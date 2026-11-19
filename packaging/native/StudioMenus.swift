import AppKit

/// Compose native responder menus; shortcuts work without a DOM imitation of OS chrome.
func installStudioMenus(_ window: StudioWindow) {
    let menu = NSMenu()
    let application = NSMenuItem()
    menu.addItem(application)
    let appMenu = NSMenu(title: "ag-media-studio")
    application.submenu = appMenu
    appMenu.addItem(withTitle: "About ag-media-studio", action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
    appMenu.addItem(.separator())
    appMenu.addItem(withTitle: "Hide ag-media-studio", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
    appMenu.addItem(.separator())
    appMenu.addItem(withTitle: "Quit ag-media-studio", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
    let edit = NSMenuItem()
    menu.addItem(edit)
    edit.submenu = NSMenu(title: "Edit")
    for (title, action, key) in [("Undo", "undo:", "z"), ("Cut", "cut:", "x"), ("Copy", "copy:", "c"), ("Paste", "paste:", "v"), ("Select All", "selectAll:", "a")] {
        edit.submenu?.addItem(withTitle: title, action: Selector(action), keyEquivalent: key)
    }
    let view = NSMenuItem()
    menu.addItem(view)
    view.submenu = NSMenu(title: "View")
    let show = NSMenuItem(title: "Show Studio", action: #selector(StudioWindow.show), keyEquivalent: "0")
    show.target = window
    view.submenu?.addItem(show)
    let sidebar = NSMenuItem(title: "Toggle Sidebar", action: #selector(StudioWindow.toggleSidebar), keyEquivalent: "s")
    sidebar.keyEquivalentModifierMask = [.command, .shift]
    sidebar.target = window
    view.submenu?.addItem(sidebar)
    for (title, action, key) in [("Reload", #selector(StudioWindow.reload), "r"), ("Open in Browser", #selector(StudioWindow.openBrowser), ""), ("Open Data Folder", #selector(StudioWindow.openData), ""), ("Open Models Folder", #selector(StudioWindow.openModels), "")] {
        let item = NSMenuItem(title: title, action: action, keyEquivalent: key)
        item.target = window
        view.submenu?.addItem(item)
    }
    let windows = NSMenuItem()
    menu.addItem(windows)
    windows.submenu = NSMenu(title: "Window")
    windows.submenu?.addItem(withTitle: "Close Window", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
    windows.submenu?.addItem(withTitle: "Minimize", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
    windows.submenu?.addItem(withTitle: "Zoom", action: #selector(NSWindow.performZoom(_:)), keyEquivalent: "")
    NSApp.windowsMenu = windows.submenu
    let help = NSMenuItem()
    menu.addItem(help)
    let helpMenu = NSMenu(title: "Help")
    help.submenu = helpMenu
    helpMenu.addItem(withTitle: "Studio Documentation", action: #selector(StudioHelp.openDocumentation), keyEquivalent: "").target = window.help
    helpMenu.addItem(withTitle: "Check for Updates…", action: #selector(StudioHelp.checkForUpdates), keyEquivalent: "").target = window.help
    NSApp.helpMenu = helpMenu
    NSApp.mainMenu = menu
}
