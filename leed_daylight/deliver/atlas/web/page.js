/* Tabs and plan layers of the floor atlas page (page.py). No library, no network.
 *
 * A button [data-tab] shows the section [data-panel] with the same value and
 * hides the others; the address #<value> opens that tab. A select
 * [data-plan-select] picks the plan layer, on every storey at once: each
 * storey shows the figure [data-layer] with the chosen key.
 */
(function () {
  "use strict";
  var tabs = Array.prototype.slice.call(document.querySelectorAll("[data-tab]"));
  var panels = Array.prototype.slice.call(document.querySelectorAll("[data-panel]"));

  function show(id) {
    var found = tabs.some(function (t) { return t.dataset.tab === id; });
    if (!found) return false;
    tabs.forEach(function (t) { t.setAttribute("aria-selected", String(t.dataset.tab === id)); });
    panels.forEach(function (p) { p.hidden = p.dataset.panel !== id; });
    return true;
  }
  tabs.forEach(function (t) {
    t.addEventListener("click", function () {
      if (!show(t.dataset.tab)) return;
      if (location.hash !== "#" + t.dataset.tab) location.hash = t.dataset.tab;
    });
  });
  function fromHash() { return show(decodeURIComponent(location.hash.replace(/^#/, ""))); }
  window.addEventListener("hashchange", fromHash);
  if (!fromHash() && tabs.length) show(tabs[0].dataset.tab);

  var selects = Array.prototype.slice.call(document.querySelectorAll("select[data-plan-select]"));
  function layer(key) {
    selects.forEach(function (s) { s.value = key; });
    Array.prototype.forEach.call(document.querySelectorAll("figure[data-layer]"), function (f) {
      f.hidden = f.dataset.layer !== key;
    });
  }
  selects.forEach(function (s) {
    s.addEventListener("change", function () { layer(s.value); });
  });
})();
