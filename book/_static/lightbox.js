// Click-to-enlarge images: a full-window overlay showing the clicked image at its largest
// legible size. Click anywhere or press Escape to close. Event delegation on the document
// keeps it working across the site's client-side page navigation.
// Injected into every built page by tools/inject_static.py.
(function () {
  function close() {
    var box = document.getElementById("dwibook-lightbox");
    if (box) box.remove();
  }
  document.addEventListener("click", function (e) {
    if (document.getElementById("dwibook-lightbox")) {
      close();
      return;
    }
    var img = e.target && e.target.closest ? e.target.closest("main img") : null;
    if (!img || img.closest("a")) return;
    var overlay = document.createElement("div");
    overlay.id = "dwibook-lightbox";
    var big = document.createElement("img");
    big.src = img.currentSrc || img.src;
    big.alt = img.alt || "";
    overlay.appendChild(big);
    document.body.appendChild(overlay);
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") close();
  });
})();
