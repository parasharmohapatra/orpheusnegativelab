import type { Mat3 } from '../types';
import { PRIMARIES, rgbToRgb, type Primaries } from './matrices';
import { TRANSFER, type TransferId } from './transfer';

/** The internal working space: linear Rec.2020 primaries, D65. Wide enough to keep film colours positive. */
export const WORKING = PRIMARIES.rec2020;

export type OutputSpaceId = 'srgb' | 'displayP3' | 'adobeRGB' | 'proPhoto';

export interface OutputSpace {
  id: OutputSpaceId;
  label: string;
  primaries: Primaries;
  transfer: TransferId;
}

export const OUTPUT_SPACES: Record<OutputSpaceId, OutputSpace> = {
  srgb: { id: 'srgb', label: 'sRGB', primaries: PRIMARIES.rec709, transfer: TRANSFER.srgb },
  displayP3: { id: 'displayP3', label: 'Display P3', primaries: PRIMARIES.displayP3, transfer: TRANSFER.srgb },
  adobeRGB: { id: 'adobeRGB', label: 'Adobe RGB (1998)', primaries: PRIMARIES.adobeRGB, transfer: TRANSFER.gamma22 },
  proPhoto: { id: 'proPhoto', label: 'ProPhoto RGB', primaries: PRIMARIES.proPhoto, transfer: TRANSFER.gamma18 },
};

export const REC709_TO_WORKING: Mat3 = rgbToRgb(PRIMARIES.rec709, WORKING);
export const WORKING_TO_REC709: Mat3 = rgbToRgb(WORKING, PRIMARIES.rec709);

export function workingToOutput(id: OutputSpaceId): Mat3 {
  return rgbToRgb(WORKING, OUTPUT_SPACES[id].primaries);
}
