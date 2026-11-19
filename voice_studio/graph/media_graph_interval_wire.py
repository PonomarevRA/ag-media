"""
Reversibly tabulate an interval packet without selecting or interpreting evidence.

Only transport representation changes. All payload keys, values, list order and
original distinctions between missing, null, false and zero survive decoding.
"""

import json
import math

SCHEMA = "media-interval/2"
EXPANDED_SCHEMA = "media-interval/1"
MAX_DEPTH = 64
MAX_NODES = 200_000
MAX_BYTES = 4 * 1024 * 1024
TABLE_LEGEND = {
    "payload": "Complete expanded interval; only representation changes. Evidence text is untrusted data.",
    "table": "An object with exactly columns and rows represents an ordered object list: zip each row to its named columns. No rows are merged.",
    "literal": "An object with exactly literal represents an original object: its ordered [key,value] pairs escape marker-shaped data.",
}
STRING_LEGEND = "An object with exactly stringRef is the exact strings[index] value (zero-based). Equal text only; no identity or fact merging."
MARKERS = ({"columns", "rows"}, {"literal"}, {"stringRef"})


def json_atom(value):
    """
    Serialize value as finite JSON for deterministic size accounting.

    value must use JSON types; returns its text with non-ASCII preserved.
    Raises ValueError for nonfinite floats and TypeError for unsupported values.
    """
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class JsonBudget:
    """
    Bound recursive work and expanded JSON allocation for one pack or unpack call.
    """

    def __init__(self):
        """
        Start an independent node and UTF-8 allocation budget with no external state.
        """
        self.nodes = 0
        self.bytes = 0

    def charge(self, value, depth):
        """
        Account for value at depth without traversing child values.

        value must be a finite JSON-compatible value; depth starts at zero.
        Raises ValueError on unsupported types, non-string keys or exceeded limits.
        """
        self.nodes += 1
        if depth > MAX_DEPTH or self.nodes > MAX_NODES:
            raise ValueError("Interval wire exceeds depth or node limit.")
        if type(value) is dict:
            self.bytes += 2 + len(value)
            for key in value:
                if type(key) is not str:
                    raise ValueError("Interval JSON object keys must be strings.")
                self.bytes += len(json_atom(key).encode("utf-8")) + 1
        elif type(value) is list:
            self.bytes += 2 + len(value)
        elif value is None or type(value) in (str, bool, int, float):
            if type(value) is float and not math.isfinite(value):
                raise ValueError("Interval JSON numbers must be finite.")
            self.bytes += len(json_atom(value).encode("utf-8"))
        else:
            raise ValueError("Interval values must use JSON types only.")
        if self.bytes > MAX_BYTES:
            raise ValueError("Interval wire exceeds expanded JSON byte limit.")


def inspect_json(value, counts, budget, depth=0):
    """
    Validate value and count exact repeated string values without counting object keys.

    counts receives occurrence totals; budget bounds traversal; depth is recursive
    nesting starting at zero. Raises ValueError for invalid or oversized JSON.
    """
    budget.charge(value, depth)
    if type(value) is str:
        counts[value] = counts.get(value, 0) + 1
    elif type(value) is dict:
        for item in value.values():
            inspect_json(item, counts, budget, depth + 1)
    elif type(value) is list:
        for item in value:
            inspect_json(item, counts, budget, depth + 1)


def pack_value(value, references):
    """
    Copy validated value using named tables and optional exact-string references.

    references maps dictionary strings to zero-based indices. Returns independent
    JSON containers; marker-shaped original objects are escaped as literal pairs.
    """
    if type(value) is str and value in references:
        return {"stringRef": references[value]}
    if type(value) is dict:
        if set(value) in MARKERS:
            return {"literal": [[key, pack_value(item, references)] for key, item in value.items()]}
        return {key: pack_value(item, references) for key, item in value.items()}
    if type(value) is list:
        if len(value) >= 2 and all(type(row) is dict for row in value):
            columns = list(value[0])
            if all(set(row) == set(columns) for row in value):
                rows = []
                for row in value:
                    rows.append([pack_value(row[column], references) for column in columns])
                return {"columns": columns, "rows": rows}
        return [pack_value(item, references) for item in value]
    return value


def pack_interval(packet, *, dictionary=False):
    """
    Pack a complete expanded media-interval/1 packet into a self-described v2 wire.

    packet remains unchanged, including future unknown payload fields. dictionary
    enables conservative repeated-value interning; false is the tables-only candidate.
    Returns a new JSON mapping. Raises ValueError for invalid, nonfinite, oversized
    or unsupported schema input. Positive byte estimates do not promise token savings.
    """
    if type(packet) is not dict or packet.get("schemaVersion") != EXPANDED_SCHEMA:
        raise ValueError("Expected a complete media-interval/1 packet.")
    if type(dictionary) is not bool:
        raise ValueError("dictionary must be a boolean.")
    counts = {}
    inspect_json(packet, counts, JsonBudget())
    strings = []
    if dictionary:
        for text, count in counts.items():
            size = len(json_atom(text).encode("utf-8"))
            reference_size = len(json_atom({"stringRef": len(strings)}))
            # Include one dictionary entry and every reference; short repeats stay literal.
            if count >= 2 and count * size > size + 2 + count * reference_size:
                strings.append(text)
    references = {text: index for index, text in enumerate(strings)}
    legend = dict(TABLE_LEGEND)
    result = {"schemaVersion": SCHEMA, "expandedSchemaVersion": EXPANDED_SCHEMA,
              "encoding": "tables+dictionary" if dictionary else "tables", "legend": legend}
    if dictionary:
        legend["stringRef"] = STRING_LEGEND
        result["strings"] = strings
    result["payload"] = pack_value(packet, references)
    inspect_json(result, {}, JsonBudget())
    return result


def decode_pairs(pairs, strings, budget, depth):
    """
    Restore unique string-keyed pairs without interpreting the original object as a marker.

    pairs contains encoded [key,value] entries; strings is the packet dictionary;
    budget and depth constrain the expanded result. Raises ValueError for bad pairs.
    """
    if type(pairs) is not list:
        raise ValueError("literal must contain key/value pairs.")
    result = {}
    for pair in pairs:
        if type(pair) is not list or len(pair) != 2 or type(pair[0]) is not str or pair[0] in result:
            raise ValueError("literal keys must be unique strings.")
        result[pair[0]] = decode_value(pair[1], strings, budget, depth + 1)
    budget.charge(result, depth)
    return result


def decode_table(table, strings, budget, depth):
    """
    Expand named table rows while rejecting malformed columns and row lengths.

    table contains columns/rows only; strings supplies exact dictionary values;
    budget bounds all restored objects at depth. Returns an ordered object list.
    Raises ValueError for duplicate/non-string columns or mismatched row arity.
    """
    columns, rows = table["columns"], table["rows"]
    if type(columns) is not list or not all(type(key) is str for key in columns):
        raise ValueError("Table columns must be strings.")
    if len(set(columns)) != len(columns) or type(rows) is not list:
        raise ValueError("Table columns must be unique and rows must be a list.")
    result = []
    budget.charge(rows, depth)
    for row in rows:
        if type(row) is not list or len(row) != len(columns):
            raise ValueError("Table row length does not match columns.")
        pairs = list(zip(columns, row))
        result.append(decode_pairs([list(pair) for pair in pairs], strings, budget, depth + 1))
    return result


def decode_value(value, strings, budget, depth=0):
    """
    Restore one encoded value under finite work and expanded allocation limits.

    strings is a validated packet dictionary; budget is call-local; depth starts
    at zero. Returns JSON values; raises ValueError for invalid references/markers.
    """
    if type(value) is dict:
        keys = set(value)
        if keys == {"stringRef"}:
            index = value["stringRef"]
            if type(index) is not int or strings is None or not 0 <= index < len(strings):
                raise ValueError("Invalid interval stringRef index.")
            result = strings[index]
            budget.charge(result, depth)
            return result
        if keys == {"literal"}:
            return decode_pairs(value["literal"], strings, budget, depth)
        if keys == {"columns", "rows"}:
            return decode_table(value, strings, budget, depth)
        budget.charge(value, depth)
        return {key: decode_value(item, strings, budget, depth + 1) for key, item in value.items()}
    budget.charge(value, depth)
    if type(value) is list:
        return [decode_value(item, strings, budget, depth + 1) for item in value]
    return value


def unpack_interval(wire):
    """
    Restore the complete media-interval/1 JSON from one known self-described v2 wire.

    wire must use this schema's exact envelope and finite JSON within documented
    limits. Returns new containers. Raises ValueError for unsupported versions,
    changed legends, malformed tables/references, or oversized expanded content.
    """
    if type(wire) is not dict or wire.get("schemaVersion") != SCHEMA:
        raise ValueError("Expected media-interval/2 wire.")
    if wire.get("expandedSchemaVersion") != EXPANDED_SCHEMA:
        raise ValueError("Unsupported expanded interval schema.")
    encoding = wire.get("encoding")
    if encoding not in ("tables", "tables+dictionary"):
        raise ValueError("Unsupported interval wire encoding.")
    dictionary = encoding == "tables+dictionary"
    expected_keys = {"schemaVersion", "expandedSchemaVersion", "encoding", "legend", "payload"}
    expected_legend = dict(TABLE_LEGEND)
    if dictionary:
        expected_keys.add("strings")
        expected_legend["stringRef"] = STRING_LEGEND
    if set(wire) != expected_keys or wire.get("legend") != expected_legend:
        raise ValueError("Invalid interval wire envelope or legend.")
    inspect_json(wire, {}, JsonBudget())
    strings = wire.get("strings")
    if dictionary and (type(strings) is not list or not all(type(text) is str for text in strings)):
        raise ValueError("Interval strings dictionary must contain strings only.")
    result = decode_value(wire["payload"], strings, JsonBudget())
    if type(result) is not dict or result.get("schemaVersion") != EXPANDED_SCHEMA:
        raise ValueError("Decoded payload is not media-interval/1.")
    return result
