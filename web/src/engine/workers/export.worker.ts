/// <reference lib="webworker" />
import { exportFrame } from '../export/exportFrame';
import type { ExportRequest, ExportResponse } from './protocol';

const post = (msg: ExportResponse, transfer: Transferable[] = []) =>
  (self as DedicatedWorkerGlobalScope).postMessage(msg, transfer);

self.onmessage = async (e: MessageEvent<ExportRequest>) => {
  const { id, job } = e.data;
  try {
    const result = await exportFrame(job, (progress) => post({ id, type: 'progress', progress }));
    post({ id, type: 'done', result }, [result.bytes.buffer]);
  } catch (err) {
    post({ id, type: 'error', error: err instanceof Error ? err.message : String(err) });
  }
};
