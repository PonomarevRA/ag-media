"""Expose the named read operations of the local media graph to the Studio UI."""

from voice_studio.graph.media_graph_store import MediaGraphStore


def read_graph(root, request):
    """Dispatch only bounded graph reads. All identifiers refer to saved snapshots."""
    actions = {
        "list": ((), ("limit", "offset")),
        "overview": (("snapshot_id",), ()),
        "search": (("snapshot_id", "query"), ("kind", "limit", "after")),
        "timeline": (("snapshot_id", "start_ms", "end_ms"), ("kind", "limit", "after")),
        "node": (("snapshot_id", "node_id"), ("limit", "offset")),
    }
    action, args = request.get("action"), request.get("args")
    if not isinstance(action, str) or action not in actions or not isinstance(args, dict):
        raise ValueError("Выбери операцию чтения графа.")
    required, optional = actions[action]
    if not set(required).issubset(args) or set(args).difference(required + optional):
        raise ValueError("Уточни параметры запроса к графу.")
    for key, value in args.items():
        if key in ("start_ms", "end_ms", "limit", "offset", "after"):
            if type(value) is not int or not 0 <= value <= 9_007_199_254_740_991:
                raise ValueError("Ожидалось целое число в параметрах графа.")
        elif not isinstance(value, str) or not 0 < len(value) <= 300:
            raise ValueError("Ожидался текстовый параметр графа длиной до 300 знаков.")

    store = MediaGraphStore(root)
    operations = {"list": store.list, "overview": store.overview, "search": store.search,
                  "timeline": store.timeline, "node": store.node}
    arguments = dict(args)
    if "snapshot_id" in arguments:
        arguments["identity"] = arguments.pop("snapshot_id")
    return operations[action](**arguments)
