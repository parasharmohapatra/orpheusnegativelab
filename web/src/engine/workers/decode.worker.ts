/// <reference lib="webworker" />
import { decodeForEditing } from '../decode/index';
import type { DecodeRequest, DecodeResponse } from './protocol';

self.onmessage = async (e: MessageEvent<DecodeRequest>) => {
  const { id, file, tiffTransfer } = e.data;
  try {
    const frame = await decodeForEditing(file, { tiffTransfer });
    const msg: DecodeResponse = { id, ok: true, frame };
    (self as DedicatedWorkerGlobalScope).postMessage(msg, [frame.proxy.data.buffer, frame.analysis.data.buffer]);
  } catch (err) {
    const msg: DecodeResponse = { id, ok: false, error: err instanceof Error ? err.message : String(err) };
    (self as DedicatedWorkerGlobalScope).postMessage(msg);
  }
};
