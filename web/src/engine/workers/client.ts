import type { DecodedFrame } from '../types';
import type { TiffTransfer } from '../decode/tiff';
import type { ExportJob, ExportResult } from '../export/exportFrame';
import type { DecodeRequest, DecodeResponse, ExportRequest, ExportResponse } from './protocol';

let nextId = 1;

interface Slot {
  worker: Worker;
  busy: boolean;
}

/**
 * A small pool of decode workers. Each holds its own LibRaw instance; big files get
 * a worker to themselves by virtue of the queue being FIFO with few slots.
 */
class DecodePool {
  private slots: Slot[] = [];
  private queue: { req: DecodeRequest; resolve: (f: DecodedFrame) => void; reject: (e: Error) => void }[] = [];
  private pending = new Map<number, { resolve: (f: DecodedFrame) => void; reject: (e: Error) => void; slot: Slot }>();

  constructor(size: number) {
    for (let i = 0; i < size; i++) {
      const worker = new Worker(new URL('./decode.worker.ts', import.meta.url), { type: 'module' });
      const slot: Slot = { worker, busy: false };
      worker.onmessage = (e: MessageEvent<DecodeResponse>) => {
        const p = this.pending.get(e.data.id);
        if (!p) return;
        this.pending.delete(e.data.id);
        slot.busy = false;
        if (e.data.ok) p.resolve(e.data.frame);
        else p.reject(new Error(e.data.error));
        this.pump();
      };
      worker.onerror = (e) => {
        for (const [id, p] of this.pending) {
          if (p.slot !== slot) continue;
          this.pending.delete(id);
          p.reject(new Error(e.message || 'Decoder crashed'));
        }
        slot.busy = false;
        this.pump();
      };
      this.slots.push(slot);
    }
  }

  decode(file: File, tiffTransfer?: TiffTransfer, priority = false): Promise<DecodedFrame> {
    return new Promise((resolve, reject) => {
      const item = { req: { id: nextId++, file, tiffTransfer }, resolve, reject };
      if (priority) this.queue.unshift(item);
      else this.queue.push(item);
      this.pump();
    });
  }

  private pump(): void {
    for (const slot of this.slots) {
      if (slot.busy || !this.queue.length) continue;
      const item = this.queue.shift()!;
      slot.busy = true;
      this.pending.set(item.req.id, { resolve: item.resolve, reject: item.reject, slot });
      slot.worker.postMessage(item.req);
    }
  }
}

let pool: DecodePool | null = null;
export function decodeFile(file: File, tiffTransfer?: TiffTransfer, priority = false): Promise<DecodedFrame> {
  pool ??= new DecodePool(Math.max(1, Math.min(3, (navigator.hardwareConcurrency || 4) - 1)));
  return pool.decode(file, tiffTransfer, priority);
}

let exportWorker: Worker | null = null;
const exportPending = new Map<number, { resolve: (r: ExportResult) => void; reject: (e: Error) => void; onProgress?: (f: number) => void }>();

/** Exports run one at a time in a dedicated worker (each can need 1–2 GB at full res). */
export function exportInWorker(job: ExportJob, onProgress?: (f: number) => void): Promise<ExportResult> {
  if (!exportWorker) {
    exportWorker = new Worker(new URL('./export.worker.ts', import.meta.url), { type: 'module' });
    exportWorker.onmessage = (e: MessageEvent<ExportResponse>) => {
      const p = exportPending.get(e.data.id);
      if (!p) return;
      if (e.data.type === 'progress') p.onProgress?.(e.data.progress);
      else {
        exportPending.delete(e.data.id);
        if (e.data.type === 'done') p.resolve(e.data.result);
        else p.reject(new Error(e.data.error));
      }
    };
    exportWorker.onerror = (e) => {
      for (const [, p] of exportPending) p.reject(new Error(e.message || 'Export worker crashed'));
      exportPending.clear();
      exportWorker = null;
    };
  }
  const id = nextId++;
  return new Promise((resolve, reject) => {
    exportPending.set(id, { resolve, reject, onProgress });
    const req: ExportRequest = { id, job };
    exportWorker!.postMessage(req);
  });
}
