import {attachStudioIcon} from "./studio-icons.js";

/** Share one navigation between the fixed sidebar and a native mobile dialog. */
export function mountStudioNavigation(find) {
  const desktop = find("desktop-navigation");
  const navigation = find("studio-navigation");
  const drawer = find("mobile-navigation");
  const slot = find("mobile-navigation-slot");
  const trigger = find("nav-open");
  const compact = matchMedia("(max-width: 52rem)");
  let outsidePress = null;
  let releasedOutside = false;

  /** Measure backdrop events because a touch release may retain implicit pointer capture. */
  function isOutside({clientX, clientY}) {
    const {left, right, top, bottom} = drawer.getBoundingClientRect();
    return clientX < left || clientX >= right || clientY < top || clientY >= bottom;
  }

  /** Move the single navigation without duplicating IDs or tab stops. */

  function placeNavigation() {
    if (compact.matches) slot.append(navigation);
    else {
      if (drawer.open) drawer.close();
      desktop.append(navigation);
    }
  }

  trigger.addEventListener("click", () => {
    outsidePress = null;
    releasedOutside = false;
    drawer.showModal();
    trigger.setAttribute("aria-expanded", "true");
    navigation.querySelector('[aria-current="page"]')?.focus();
  });
  find("nav-close").addEventListener("click", () => drawer.close());
  drawer.addEventListener("close", () => trigger.setAttribute("aria-expanded", "false"));
  /** Treat movement across the drawer edge or more than 8 CSS px as a drag. */
  function qualifies(event) {
    return outsidePress?.id === event.pointerId && isOutside(event) &&
      Math.hypot(event.clientX - outsidePress.x, event.clientY - outsidePress.y) <= 8;
  }

  drawer.addEventListener("pointerdown", event => {
    releasedOutside = false;
    outsidePress = event.isPrimary && event.button === 0 && isOutside(event)
      ? {id: event.pointerId, x: event.clientX, y: event.clientY} : null;
  });
  drawer.addEventListener("pointermove", event => {
    if (outsidePress && !qualifies(event)) outsidePress = null;
  });
  drawer.addEventListener("pointerup", event => {
    releasedOutside = qualifies(event);
    outsidePress = null;
  });
  drawer.addEventListener("pointercancel", () => { outsidePress = null; releasedOutside = false; });
  drawer.addEventListener("click", event => {
    if (releasedOutside && event.target === drawer && event.detail) drawer.close();
    releasedOutside = false;
  });
  compact.addEventListener("change", placeNavigation);
  placeNavigation();
  for (const command of document.querySelectorAll("[data-icon]")) attachStudioIcon(command);

  return {
    /** Mark the current link and close the drawer before the router focuses the heading. */
    select(id) {
      navigation.querySelector("[aria-current]")?.removeAttribute("aria-current");
      const link = find(id);
      link.setAttribute("aria-current", "page");
      find("studio-current-page").textContent = link.textContent;
      if (drawer.open) drawer.close();
    },
  };
}
