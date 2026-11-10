//! Opt-in stage timing for native builds (`PHONON_PROF=1`); compiled to a plain call on wasm.

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    use std::collections::BTreeMap;
    use std::sync::Mutex;
    use std::time::Instant;

    static TABLE: Mutex<BTreeMap<&'static str, (f64, u64)>> = Mutex::new(BTreeMap::new());

    fn enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("PHONON_PROF").is_some())
    }

    pub fn time<R>(name: &'static str, f: impl FnOnce() -> R) -> R {
        if !enabled() {
            return f();
        }
        let t = Instant::now();
        let r = f();
        let dt = t.elapsed().as_secs_f64();
        let mut tab = TABLE.lock().unwrap();
        let e = tab.entry(name).or_insert((0.0, 0));
        e.0 += dt;
        e.1 += 1;
        r
    }

    /// Print and clear the table.
    pub fn report() {
        if !enabled() {
            return;
        }
        let mut tab = TABLE.lock().unwrap();
        let total: f64 = tab.values().map(|v| v.0).sum();
        let mut rows: Vec<_> = tab.iter().collect();
        rows.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
        for (k, (s, n)) in rows {
            eprintln!("  {k:<14} {s:>8.3}s {:>5.1}%  x{n}", 100.0 * s / total);
        }
        tab.clear();
    }
}

#[cfg(target_arch = "wasm32")]
mod imp {
    #[inline(always)]
    pub fn time<R>(_: &'static str, f: impl FnOnce() -> R) -> R {
        f()
    }
    pub fn report() {}
}

pub use imp::{report, time};
