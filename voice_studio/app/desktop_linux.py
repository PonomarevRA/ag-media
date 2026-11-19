"""Native GTK3/WebKit2 desktop host; packaged Linux uses explicit distro GUI dependencies."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
import webbrowser

from voice_paths import application_data


def identity(pid):
    """Read a Linux process's parent and kernel start tick, retaining PID-reuse protection."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        return int(fields[1]), fields[19]
    except (OSError, ValueError, IndexError):
        return None


class LinuxOwner:
    """Track descendants by kernel identity, including children in separate process groups."""

    def __init__(self, process):
        """Start tracking only the directly launched backend and its future descendants."""
        self.process = process
        self.root = identity(process.pid)
        self.children = {}
        self.guard = threading.Lock()
        self.finished = threading.Event()
        threading.Thread(target=self.monitor, name="native-process-owner", daemon=True).start()

    def sample(self):
        """Capture child relationships while the launched service remains the same process."""
        if not self.same_process(self.process.pid, self.root):
            return
        pending = [self.process.pid]
        visited = set()
        while pending:
            parent = pending.pop()
            if parent in visited:
                continue
            visited.add(parent)
            try:
                task_files = list(Path(f"/proc/{parent}/task").glob("*/children"))
                children = set()
                for path in task_files:
                    children.update(int(value) for value in path.read_text().split())
            except (OSError, ValueError):
                continue
            for child in children:
                value = identity(child)
                if value and value[0] == parent:
                    self.children[child] = value
                    pending.append(child)

    @staticmethod
    def same_process(pid, observed):
        """Compare kernel start time, retaining ownership even after a worker is reparented."""
        current = identity(pid)
        return current is not None and observed is not None and current[1] == observed[1]

    def monitor(self):
        """Observe owned workers without polling the web UI or claiming unrelated processes."""
        while not self.finished.wait(.2):
            with self.guard:
                self.sample()

    def stop(self):
        """Drain on a background thread, then kill only still-matching owned identities after six seconds."""
        self.finished.set()
        with self.guard:
            self.sample()
            owned = dict(self.children)
        owned[self.process.pid] = self.root
        try:
            self.process.stdin.write("quit\n")
            self.process.stdin.flush()
        except (OSError, ValueError):
            pass
        for pid, value in owned.items():
            if self.same_process(pid, value):
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            if all(not self.same_process(pid, value) for pid, value in owned.items()):
                break
            time.sleep(.05)
        for pid, value in owned.items():
            if self.same_process(pid, value):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.process.wait(timeout=3)


def main():
    """Run a GTK window for an owned service; ordinary CLI MCP/config modes open no GUI."""
    root = Path(__file__).resolve().parents[2]
    libraries = root.parent / "python"
    if libraries.is_dir():
        sys.path.insert(0, str(libraries))
    if "--mcp" in sys.argv or "--print-config" in sys.argv:
        from voice_studio.app.desktop_config import main as config_main
        config_main()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("VOICE_STUDIO_DATA", application_data() / "studio")))
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--demo", action="store_true")
    options = parser.parse_args()
    data = options.data.expanduser().resolve()
    data.mkdir(mode=0o700, parents=True, exist_ok=True)
    import gi
    gi.require_version("Gtk", "3.0")
    gi.require_version("WebKit2", "4.1")
    from gi.repository import Gtk, GLib, WebKit2, Gdk
    # Scope desktop activation to the selected workspace, before starting another backend.
    app_id = "org.agmediastudio.workspace" + hashlib.sha256(str(data).encode()).hexdigest()[:24]
    application = Gtk.Application(application_id=app_id)
    application.register(None)
    if application.get_is_remote():
        application.activate()
        return
    application.hold()
    bootstrap = "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));sys.path.insert(0,sys.argv.pop(1));runpy.run_module('voice_studio.app.desktop_runtime',run_name='__main__')"
    command = [sys.executable, "-I", "-B", "-c", bootstrap, str(libraries), str(root), "--data", str(data), "--port", str(options.port)]
    if options.demo:
        command.append("--demo")
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}}
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    log = (data / "desktop.log").open("a")
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, cwd=data, env=environment, start_new_session=True)
    owner = LinuxOwner(process)
    window = Gtk.Window(title="ag-media-studio")
    window.set_icon_from_file(str(root / "web/brand/icon-512.png"))
    window.set_application(application)
    window.set_default_size(1280, 840)
    window.set_size_request(390, 560)
    header = Gtk.HeaderBar(title="ag-media-studio", show_close_button=True)
    window.set_titlebar(header)
    content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    window.add(content)
    message = Gtk.Label(label="Starting local Studio…")
    content.pack_start(message, True, True, 0)
    state_file = data / "desktop-window.json"
    try:
        preferences = json.loads(state_file.read_text())
    except (OSError, ValueError):
        preferences = {}
    if all(type(preferences.get(key)) is int for key in ("width", "height")):
        window.resize(max(390, preferences["width"]), max(560, preferences["height"]))
    active = {"origin": None, "view": None, "quitting": False, "hidden": False, "checking_updates": False}

    def persist():
        """Save only native size and the navigation presentation flag beside the workspace."""
        width, height = window.get_size()
        preferences.update(width=width, height=height)
        state_file.write_text(json.dumps(preferences) + "\n")

    def quit_app(*_):
        """Keep GTK responsive while its exact owned process tree drains in the background."""
        if active["quitting"]:
            return True
        active["quitting"] = True
        persist()
        window.set_sensitive(False)

        def drain():
            """Finish the GUI only after the bounded owned-process cleanup returns."""
            try:
                owner.stop()
            finally:
                log.close()
                GLib.idle_add(application.quit)
        threading.Thread(target=drain, name="native-shutdown", daemon=True).start()
        return True

    def hide_app(*_):
        """Retain the current WebView and service when the window-manager close button is used."""
        persist()
        active["hidden"] = True
        window.hide()
        return True

    def show_app(*_):
        """Restore the existing document on desktop activation, without loading a second workspace."""
        if active["quitting"]:
            return
        active["hidden"] = False
        window.show_all()
        window.present()

    def sidebar(*_):
        """Trigger only the existing human sidebar action in the owned document."""
        if active["view"]:
            active["view"].run_javascript("document.getElementById('nav-collapse')?.click()", None, None, None)

    def browser(*_):
        """Retain the system-browser fallback for capabilities missing in WebKitGTK."""
        if active["origin"]:
            webbrowser.open(active["origin"])

    def documentation(*_):
        """Open the owned Wiki without changing the active document."""
        if active["origin"]:
            webbrowser.open(active["origin"] + "/wiki")

    def show_updates(result):
        """Show version status in the native window and let the user open the releases page."""
        active["checking_updates"] = False
        if active["quitting"] or active["hidden"]:
            return False
        dialog = Gtk.MessageDialog(transient_for=window, flags=Gtk.DialogFlags.DESTROY_WITH_PARENT,
                                   message_type=Gtk.MessageType.INFO, buttons=Gtk.ButtonsType.OK,
                                   text="AG Media Studio Updates")
        dialog.format_secondary_text(result["message"])
        url = result.get("releasesUrl")
        if url:
            dialog.add_button("Open Releases", Gtk.ResponseType.APPLY)
        decision = dialog.run()
        dialog.destroy()
        if decision == Gtk.ResponseType.APPLY and url:
            webbrowser.open(url)
        return False

    def check_updates(*_):
        """Start one check on demand while GTK continues processing the active document."""
        if active["checking_updates"] or active["quitting"]:
            return
        active["checking_updates"] = True

        def check():
            """Send the shared release result back to the GTK event loop."""
            from voice_studio.app.desktop_updates import check_desktop_update
            GLib.idle_add(show_updates, check_desktop_update())

        threading.Thread(target=check, name="studio-update-check", daemon=True).start()

    for label, title, callback in (("Toggle Sidebar (Ctrl+Shift+S)", "Sidebar", sidebar), ("Open in Browser", "Browser", browser)):
        # Text remains usable even when a minimal distro omits an icon theme or SVG loader.
        button = Gtk.Button(label=title)
        button.set_tooltip_text(label)
        button.get_accessible().set_name(label)
        button.connect("clicked", callback)
        header.pack_start(button)
    menu = Gtk.Menu()
    for label, callback in (("Toggle Sidebar", sidebar), ("Open in Browser", browser), ("Open Data Folder", lambda *_: subprocess.Popen(["xdg-open", str(data)])),
                            ("Open Models Folder", lambda *_: subprocess.Popen(["xdg-open", os.environ.get("VOICE_MODELS_DIR", str(application_data() / "Models"))])), ("Quit", quit_app)):
        item = Gtk.MenuItem(label=label)
        item.connect("activate", callback)
        menu.append(item)
    help_item = Gtk.MenuItem(label="Help")
    help_menu = Gtk.Menu()
    for label, callback in (("Studio Documentation", documentation), ("Check for Updates…", check_updates)):
        item = Gtk.MenuItem(label=label)
        item.connect("activate", callback)
        help_menu.append(item)
    help_item.set_submenu(help_menu)
    menu.append(help_item)
    menu.show_all()
    more = Gtk.MenuButton()
    more.set_image(Gtk.Image.new_from_icon_name("open-menu-symbolic", Gtk.IconSize.BUTTON))
    more.set_tooltip_text("Studio menu")
    more.set_popup(menu)
    header.pack_end(more)
    shortcuts = Gtk.AccelGroup()
    window.add_accel_group(shortcuts)
    key, modifiers = Gtk.accelerator_parse("<Control><Shift>s")
    menu.get_children()[0].add_accelerator("activate", shortcuts, key, modifiers, Gtk.AccelFlags.VISIBLE)
    window.connect("delete-event", hide_app)
    application.connect("activate", show_app)
    for number in (signal.SIGTERM, signal.SIGINT):
        GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, number, quit_app)

    def loaded(_channel, result):
        """Accept only the owned page's boolean presentation state from the fixed channel."""
        if not active["view"] or not active["origin"]:
            return
        uri = active["view"].get_uri() or ""
        if urlsplit(uri).netloc != urlsplit(active["origin"]).netloc:
            return
        try:
            value = json.loads(result.get_js_value().to_json(0))
            if type(value.get("collapsed")) is bool:
                preferences["sidebarCollapsed"] = value["collapsed"]
                persist()
        except (ValueError, TypeError):
            pass

    def policy(view, decision, kind):
        """Navigate only within the bound origin; user external links use the system browser."""
        if kind == WebKit2.PolicyDecisionType.NAVIGATION_ACTION:
            action = decision.get_navigation_action()
            uri = action.get_request().get_uri()
            target = urlsplit(uri)
            source = urlsplit(active["origin"])
            if (target.scheme, target.netloc) != (source.scheme, source.netloc) and target.scheme != "blob":
                if action.is_user_gesture() and target.scheme in {"https", "http"}:
                    webbrowser.open(uri)
                decision.ignore()
                return True
        return False

    def receive(_source, condition):
        """Trust one readiness envelope from the directly spawned Python child."""
        if active["quitting"]:
            return False
        if condition & GLib.IO_HUP:
            message.set_text("Studio could not start. See desktop.log in the data folder.")
            return False
        line = process.stdout.readline(65536)
        try:
            status = json.loads(line)
        except ValueError:
            return True
        port = status.get("port")
        if status.get("schema") != "ag-media-desktop-ready-v1" or status.get("pid") != process.pid or type(port) is not int or not 1 <= port <= 65535 or status.get("origin") != f"http://127.0.0.1:{port}":
            return True
        active["origin"] = status["origin"]
        manager = WebKit2.UserContentManager()
        manager.register_script_message_handler("studioPresentation")
        manager.connect("script-message-received::studioPresentation", loaded)
        collapsed = "1" if preferences.get("sidebarCollapsed") else "0"
        script = "try {localStorage.setItem('voice-studio:sidebar-collapsed','" + collapsed + "');} catch {} document.addEventListener('click',event=>{if(event.target.closest('#nav-collapse'))setTimeout(()=>window.webkit.messageHandlers.studioPresentation.postMessage({collapsed:document.documentElement.dataset.sidebarCollapsed==='true'}),0);});"
        manager.add_script(WebKit2.UserScript.new(script, WebKit2.UserContentInjectedFrames.TOP_FRAME, WebKit2.UserScriptInjectionTime.START, None, None))
        view = WebKit2.WebView.new_with_user_content_manager(manager)
        active["view"] = view
        view.connect("decide-policy", policy)
        content.remove(message)
        content.pack_start(view, True, True, 0)
        view.load_uri(active["origin"])
        if not active["hidden"]:
            window.show_all()
        status.update(hostPid=os.getpid(), host="GTK3/WebKit2GTK")
        if os.environ.get("VOICE_DESKTOP_STATUS_FILE"):
            Path(os.environ["VOICE_DESKTOP_STATUS_FILE"]).write_text(json.dumps(status) + "\n")
        return False

    GLib.io_add_watch(process.stdout, GLib.IO_IN | GLib.IO_HUP, receive)
    try:
        application.run([sys.argv[0]])
    finally:
        application.release()
        if not active["quitting"]:
            owner.stop()


if __name__ == "__main__":
    main()
