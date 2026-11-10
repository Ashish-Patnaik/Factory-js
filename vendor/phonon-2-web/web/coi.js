// Registers coi-sw.js when the page is not cross-origin isolated, then reloads once so the headers take effect.
// Without it (or without service workers) the app still runs, on the single-threaded engine.
if (!self.crossOriginIsolated && 'serviceWorker' in navigator && !sessionStorage.getItem('coi-reloaded')) {
  navigator.serviceWorker.register(new URL('./coi-sw.js', import.meta.url)).then((reg) => {
    const reload = () => { sessionStorage.setItem('coi-reloaded', '1'); location.reload(); };
    if (reg.active && !navigator.serviceWorker.controller) reload();
    else navigator.serviceWorker.addEventListener('controllerchange', reload, { once: true });
  }).catch(() => {});
}
