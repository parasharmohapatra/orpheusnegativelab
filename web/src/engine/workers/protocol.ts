import type { DecodedFrame } from '../types';
import type { TiffTransfer } from '../decode/tiff';
import type { ExportJob, ExportResult } from '../export/exportFrame';

export interface DecodeRequest {
  id: number;
  file: File;
  tiffTransfer?: TiffTransfer;
}

export type DecodeResponse = { id: number; ok: true; frame: DecodedFrame } | { id: number; ok: false; error: string };

export interface ExportRequest {
  id: number;
  job: ExportJob;
}

export type ExportResponse =
  | { id: number; type: 'progress'; progress: number }
  | { id: number; type: 'done'; result: ExportResult }
  | { id: number; type: 'error'; error: string };
