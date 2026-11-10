/* tslint:disable */
/* eslint-disable */

export class Phonon {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Initialise WebGL2 (an OffscreenCanvas context in this worker) and upload the encoder (once). Returns the
     * renderer description and makes WebGL the active backend.
     */
    enableGl(): string;
    /**
     * Initialise WebGPU and upload the encoder (once). Resolves to the adapter description.
     */
    enableGpu(): Promise<any>;
    /**
     * JSON: weights in memory, threads, backend.
     */
    info(): string;
    /**
     * Live dictation: feed microphone samples (16 kHz mono, any block size). Resolves to JSON
     * `{"partial": text|null, "finals": [text...], "segment_s": seconds in flight}`; superseded partials are skipped.
     */
    liveFeed(samples: Float32Array): Promise<any>;
    /**
     * End the live session: finalise the segment in flight.
     */
    liveFinish(): Promise<any>;
    /**
     * Drop the in-flight live segment.
     */
    liveReset(): void;
    /**
     * Load `phonon-2.bps.tar.zst` (or a bare `model.fermion` with `config` = the bytes of its `config.json`).
     * `progress(stage, done, total)` is called while the weights are read.
     */
    static load(bytes: Uint8Array, config?: Uint8Array | null, progress?: Function | null): Phonon;
    /**
     * Drop a GPU backend's device state and weights ("webgpu" or "webgl"), freeing its GPU memory. If it was active,
     * the CPU takes over.
     */
    release(name: string): void;
    /**
     * Transcribe 16 kHz mono audio of any length. Resolves to JSON: text, words, tokens, windows, timings.
     */
    transcribe(audio: Float32Array): Promise<any>;
    /**
     * Switch between initialised backends: "cpu", "webgpu" or "webgl".
     */
    useBackend(name: string): void;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_phonon_free: (a: number, b: number) => void;
    readonly phonon_enableGl: (a: number) => [number, number, number, number];
    readonly phonon_enableGpu: (a: number) => any;
    readonly phonon_info: (a: number) => [number, number];
    readonly phonon_liveFeed: (a: number, b: number, c: number) => any;
    readonly phonon_liveFinish: (a: number) => any;
    readonly phonon_liveReset: (a: number) => void;
    readonly phonon_load: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly phonon_release: (a: number, b: number, c: number) => void;
    readonly phonon_transcribe: (a: number, b: number, c: number) => any;
    readonly phonon_useBackend: (a: number, b: number, c: number) => [number, number];
    readonly wasm_bindgen_1dae03bd03433b25___convert__closures_____invoke___js_sys_509fdf4112609f85___Function_fn_wasm_bindgen_1dae03bd03433b25___JsValue_____wasm_bindgen_1dae03bd03433b25___sys__Undefined___js_sys_509fdf4112609f85___Function_fn_wasm_bindgen_1dae03bd03433b25___JsValue_____wasm_bindgen_1dae03bd03433b25___sys__Undefined_______true_: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen_1dae03bd03433b25___convert__closures_____invoke___wasm_bindgen_1dae03bd03433b25___JsValue__core_98e5b52de1ee3cd3___result__Result_____wasm_bindgen_1dae03bd03433b25___JsError___true_: (a: number, b: number, c: any) => [number, number];
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_destroy_closure: (a: number, b: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
