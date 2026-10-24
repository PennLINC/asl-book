// Section boxes: wrap each recurring chapter section (the heading and everything up to the
// next top-level heading) in a labelled, tinted box, see sections.css. The theme renders
// headings and content as direct children of the article, so the wrapper is a plain block in
// the article's body column. Runs after the page has hydrated and again after the site's
// client-side navigation, idempotently. Injected into every built page by tools/inject_static.py.
(function () {
  var KINDS = [
    [/^see it\b/i, "see", "See it"],
    [/^measure it\b/i, "measure", "Measure it"],
    [/^residual error\b/i, "measure", "Measure it"],
    [/^correction step by step\b/i, "fix", "Correction"],
    [/^what this implies\b/i, "implies", "Implications"],
    [/^learning goals\b/i, "goals", "Goals"],
    [/^further reading\b/i, "reading", "Reading"],
  ];
  var busy = false;

  function kindOf(h2) {
    var text = (h2.textContent || "").replace(/¶\s*$/, "").trim();
    for (var i = 0; i < KINDS.length; i++) {
      if (KINDS[i][0].test(text)) return KINDS[i];
    }
    return null;
  }

  window.aslbookSections = { runs: 0, decorate: decorate };

  function decorate() {
    window.aslbookSections.runs += 1;
    var article = document.querySelector("main article");
    if (!article || busy) return;
    busy = true;
    try {
      var kids = Array.prototype.slice.call(article.children);
      for (var i = 0; i < kids.length; i++) {
        var h2 = kids[i];
        if (h2.tagName !== "H2" || h2.parentElement !== article) continue;
        var kind = kindOf(h2);
        if (!kind) continue;
        var box = document.createElement("section");
        box.className = "aslbook-box aslbook-" + kind[1];
        h2.setAttribute("data-aslbook-label", kind[2]);
        article.insertBefore(box, h2);
        var node = h2;
        while (node && !(node !== h2 && node.tagName === "H2")) {
          var next = node.nextSibling;
          box.appendChild(node);
          node = next;
        }
      }
    } finally {
      busy = false;
    }
  }

  function schedule() {
    if (schedule.timer) clearTimeout(schedule.timer);
    schedule.timer = setTimeout(decorate, 150);
  }

  function hydrated(el) {
    // React marks the nodes it owns once hydration (or a client render) has happened
    for (var k in el) if (k.indexOf("__reactFiber") === 0) return true;
    return false;
  }

  function start() {
    // Touching the DOM before React has hydrated it makes React discard the page and render
    // it again from scratch, so wait until the article carries React's fiber marker.
    // If React has not claimed the page after a few seconds (hydration failed, or the
    // page is served without its scripts), the DOM is static and safe to edit anyway.
    var tries = 0;
    (function waitForReact() {
      var article = document.querySelector("main article");
      if ((article && hydrated(article)) || tries >= 40) {
        decorate();
        observe();
      } else {
        tries += 1;
        setTimeout(waitForReact, 100);
      }
    })();
  }

  function observe() {
    var main = document.querySelector("main") || document.body;
    new MutationObserver(function (records) {
      for (var i = 0; i < records.length; i++) {
        var t = records[i].target;
        if (t.nodeType === 1 && t.closest(".aslbook-box")) continue;
        schedule();
        return;
      }
    }).observe(main, { childList: true, subtree: true });
  }

  if (document.readyState === "complete") start();
  else window.addEventListener("load", start);
})();
