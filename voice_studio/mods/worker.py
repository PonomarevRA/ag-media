"""Disposable trusted-code JSON worker. This process is not an operating-system sandbox."""

import contextlib
import importlib.util
import json
from pathlib import Path
import sys


def load_function(filename, function, modules):
    """Import each captured package file once so model and adapters share one worker state."""
    key = str(filename.resolve())
    if key not in modules:
        name = "voice_media_user_mod_" + str(len(modules))
        specification = importlib.util.spec_from_file_location(name, filename)
        module = importlib.util.module_from_spec(specification)
        sys.modules[name] = module
        specification.loader.exec_module(module)
        modules[key] = module
    callback = getattr(modules[key], function)
    if not callable(callback):
        raise TypeError(f"{filename.name}:{function} is not callable")
    return callback


def main():
    """Read one request, import its captured local worker and emit exactly one JSON envelope."""
    sys.dont_write_bytecode = True
    try:
        request = json.load(sys.stdin)
        # Import only host-owned schema checks before adding user package paths.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from voice_studio.mods.schema import validate_value
        from voice_studio.mods.contracts import validate_contract
        filename = Path(request["entrypoint"]).resolve()
        package_root = Path(request.get("packageRoot", filename.parent))
        sys.path.insert(0, str(package_root))
        sys.path.insert(0, str(filename.parent))
        modules = {}
        # Logs belong on bounded stderr; stdout is reserved for the protocol envelope.
        with contextlib.redirect_stdout(sys.stderr):
            payload = request["payload"]
            if "prepare_input" in request:
                adapter_file, adapter_name = request["prepare_input"].split(":")
                prepare = load_function(package_root / adapter_file, adapter_name, modules)
                payload = prepare(payload, request["settings"])
            run = load_function(filename, request["function"], modules)
            result = run(payload, request["settings"])
            if "rawOutputSchema" in request:
                validate_value(result, request["rawOutputSchema"], "model raw output")
            if "convert_output" in request:
                adapter_file, adapter_name = request["convert_output"].split(":")
                convert = load_function(package_root / adapter_file, adapter_name, modules)
                result = convert(result, request["settings"])
            if "contract" in request:
                validate_contract(result, request["contract"], "provider output", capability=request["capability"])
        encoded = json.dumps({"ok": True, "result": result}, ensure_ascii=False, allow_nan=False)
        sys.stdout.write(encoded)
    except BaseException as error:
        sys.stdout.write(json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"[:2000]}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
