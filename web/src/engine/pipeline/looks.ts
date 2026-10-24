import type { CurvePoint } from './recipe';

/**
 * Film looks: colour and tone signatures inspired by popular stocks, applied on top of
 * the user's own adjustments (as offsets, scaled by the look's strength) rather than
 * replacing them. They're interpretations, not measured profiles: the conversion itself
 * still comes from the negative.
 *
 * Units match the recipe: tone/colour sliders in [-1, 1]; HSL arrays are the 8 bands
 * Red, Orange, Yellow, Green, Aqua, Blue, Purple, Magenta; curves in encoded output space.
 */
export interface FilmLook {
  id: string;
  name: string;
  family: 'Kodak' | 'Fujifilm' | 'CineStill' | 'Cinema' | 'Other' | 'Black & white';
  description: string;
  /** B&W looks only change tone; colour looks are hidden for B&W film and vice versa. */
  bw?: boolean;
  tone?: Partial<Record<'contrast' | 'highlights' | 'shadows' | 'whites' | 'blacks', number>>;
  color?: Partial<Record<'temp' | 'tint' | 'saturation' | 'vibrance', number>> & {
    hslHue?: number[];
    hslSat?: number[];
    hslLum?: number[];
  };
  curves?: Partial<Record<'rgb' | 'r' | 'g' | 'b', CurvePoint[]>>;
}

const p = (...xy: number[]): CurvePoint[] => {
  const out: CurvePoint[] = [];
  for (let i = 0; i < xy.length; i += 2) out.push({ x: xy[i], y: xy[i + 1] });
  return out;
};
//                 R     O     Y     G     A     B     P     M
const hsl = (r = 0, o = 0, y = 0, g = 0, a = 0, b = 0, pu = 0, m = 0) => [r, o, y, g, a, b, pu, m];

export const FILM_LOOKS: FilmLook[] = [
  // ---- Kodak ----
  {
    id: 'portra160',
    name: 'Portra 160',
    family: 'Kodak',
    description: 'Soft and true to life; gentle skin, quiet greens',
    tone: { contrast: -0.12, highlights: -0.15, shadows: 0.1 },
    color: { temp: 0.04, saturation: -0.08, vibrance: 0.05, hslHue: hsl(0, 0, 0, -0.1), hslSat: hsl(0, -0.05, 0, -0.2, -0.15, -0.1), hslLum: hsl(0, 0.1) },
    curves: { rgb: p(0, 0.03, 0.25, 0.24, 0.75, 0.77, 1, 0.98) },
  },
  {
    id: 'portra400',
    name: 'Portra 400',
    family: 'Kodak',
    description: 'Warm, creamy skin; olive greens; forgiving highlights',
    tone: { contrast: -0.08, highlights: -0.12, shadows: 0.06 },
    color: { temp: 0.06, tint: 0.02, saturation: -0.03, hslHue: hsl(0, 0, -0.05, -0.15, 0, -0.1), hslSat: hsl(0, 0, 0, -0.15, -0.1, -0.05), hslLum: hsl(0, 0.08) },
    curves: { rgb: p(0, 0.02, 0.5, 0.51, 1, 0.99), r: p(0, 0, 0.5, 0.515, 1, 1) },
  },
  {
    id: 'portra800',
    name: 'Portra 800',
    family: 'Kodak',
    description: 'Warmer and a touch richer; amber shadows',
    tone: { highlights: -0.1 },
    color: { temp: 0.08, saturation: 0.02, hslHue: hsl(0, 0, 0, -0.15), hslSat: hsl(0.05, 0.05, 0, -0.1), hslLum: hsl(0, 0.05) },
    curves: { r: p(0, 0.02, 0.3, 0.31, 1, 1), b: p(0, 0, 0.3, 0.28, 1, 0.98) },
  },
  {
    id: 'ektar100',
    name: 'Ektar 100',
    family: 'Kodak',
    description: 'Vivid and punchy; deep blue skies, bold reds',
    tone: { contrast: 0.15, blacks: -0.05 },
    color: { saturation: 0.15, vibrance: 0.1, hslSat: hsl(0.15, 0.05, 0, 0.1, 0.1, 0.2), hslLum: hsl(0, 0, 0, 0, 0, -0.1) },
    curves: { rgb: p(0, 0, 0.25, 0.21, 0.75, 0.8, 1, 1) },
  },
  {
    id: 'gold200',
    name: 'Gold 200',
    family: 'Kodak',
    description: 'Golden, nostalgic warmth; yellow-leaning highlights',
    tone: { contrast: 0.05 },
    color: { temp: 0.15, tint: 0.02, saturation: 0.08, hslHue: hsl(0, 0, -0.1, -0.2), hslSat: hsl(0, 0.1, 0.2, -0.05, 0, -0.1) },
    curves: { b: p(0, 0.02, 0.5, 0.47, 1, 0.95) },
  },
  {
    id: 'ultramax400',
    name: 'UltraMax 400',
    family: 'Kodak',
    description: 'Everyday consumer colour; warm and saturated',
    tone: { contrast: 0.08 },
    color: { temp: 0.08, saturation: 0.1, hslHue: hsl(0, 0, 0, -0.1), hslSat: hsl(0.1, 0, 0, 0, 0, 0.05) },
    curves: { r: p(0, 0.01, 0.5, 0.52, 1, 1) },
  },
  {
    id: 'colorplus200',
    name: 'ColorPlus 200',
    family: 'Kodak',
    description: 'Budget-film charm; yellow cast, faded blacks',
    tone: { highlights: -0.1, blacks: 0.1 },
    color: { temp: 0.12, saturation: 0.05, hslHue: hsl(0, 0, 0, -0.25), hslSat: hsl(0, 0, 0.15, -0.15) },
  },
  // ---- Fujifilm ----
  {
    id: 'pro400h',
    name: 'Pro 400H',
    family: 'Fujifilm',
    description: 'Airy pastels; minty greens, cool clean whites',
    tone: { contrast: -0.1, highlights: -0.05, shadows: 0.08 },
    color: { temp: -0.06, tint: -0.04, saturation: -0.05, hslHue: hsl(0, 0, 0, 0.2, 0, -0.1), hslSat: hsl(0, -0.1, 0, -0.05, 0.1), hslLum: hsl(0, 0.1, 0, 0.1) },
    curves: { rgb: p(0, 0.04, 1, 0.98), g: p(0, 0.015, 0.5, 0.51, 1, 1) },
  },
  {
    id: 'superia400',
    name: 'Superia 400',
    family: 'Fujifilm',
    description: 'Green-tinged shadows, crisp magenta-leaning highlights',
    tone: { contrast: 0.08 },
    color: { tint: -0.03, saturation: 0.08, hslHue: hsl(0.08, 0, 0, 0.1), hslSat: hsl(0, 0, 0, 0.15, 0, 0.1) },
    curves: { g: p(0, 0.03, 0.3, 0.31, 0.8, 0.79, 1, 0.98), r: p(0, 0, 0.8, 0.81, 1, 1) },
  },
  {
    id: 'c200',
    name: 'Fujicolor C200',
    family: 'Fujifilm',
    description: 'Cool, slightly green everyday film',
    tone: { contrast: 0.03 },
    color: { temp: -0.03, tint: -0.05, hslHue: hsl(0, 0, 0, 0.15), hslSat: hsl(0, 0, -0.1) },
    curves: { g: p(0, 0.02, 0.35, 0.36, 1, 1) },
  },
  {
    id: 'velvia50',
    name: 'Velvia 50',
    family: 'Fujifilm',
    description: 'Slide-film intensity; saturated greens and blues',
    tone: { contrast: 0.25, blacks: -0.15 },
    color: { saturation: 0.3, vibrance: 0.1, hslHue: hsl(0, 0, 0, 0.05), hslSat: hsl(0.15, 0, 0, 0.25, 0.1, 0.3), hslLum: hsl(0, 0, 0, 0, 0, -0.15) },
    curves: { rgb: p(0, 0, 0.25, 0.19, 0.75, 0.82, 1, 1) },
  },
  {
    id: 'provia100',
    name: 'Provia 100F',
    family: 'Fujifilm',
    description: 'Neutral slide film; clean and accurate with bite',
    tone: { contrast: 0.12 },
    color: { saturation: 0.08, hslSat: hsl(0, 0, 0, 0.05, 0, 0.08) },
    curves: { rgb: p(0, 0, 0.25, 0.22, 0.75, 0.78, 1, 1) },
  },
  // ---- CineStill ----
  {
    id: 'cinestill800t',
    name: 'CineStill 800T',
    family: 'CineStill',
    description: 'Tungsten film by night: teal shadows, warm glow, neon reds',
    tone: { contrast: 0.08 },
    color: { temp: -0.2, tint: -0.03, saturation: 0.05, hslHue: hsl(0, -0.1), hslSat: hsl(0.15, 0, 0, 0, 0.15, 0.1) },
    curves: { b: p(0, 0.05, 0.3, 0.33, 0.7, 0.7, 1, 0.97), g: p(0, 0.03, 0.5, 0.5, 1, 0.99), r: p(0, 0, 0.3, 0.28, 0.8, 0.82, 1, 1) },
  },
  {
    id: 'cinestill50d',
    name: 'CineStill 50D',
    family: 'CineStill',
    description: 'Clean daylight cinema stock; fine, cool, cinematic',
    tone: { contrast: 0.05, highlights: -0.15 },
    color: { temp: -0.04, saturation: 0.05, hslHue: hsl(0, 0, 0, 0, 0, -0.05), hslSat: hsl(0, 0, 0, -0.05, 0, 0.1) },
    curves: { rgb: p(0, 0.02, 0.25, 0.23, 0.75, 0.78, 1, 0.99) },
  },
  {
    id: 'cinestill400d',
    name: 'CineStill 400D',
    family: 'CineStill',
    description: 'Warm, wide-latitude daylight look',
    tone: { contrast: 0.04, highlights: -0.1 },
    color: { temp: 0.06, saturation: 0.04, hslHue: hsl(0, 0, 0, -0.1), hslSat: hsl(0, 0.08, 0, -0.1) },
    curves: { b: p(0, 0.03, 0.4, 0.4, 1, 0.97) },
  },
  // ---- Cinema ----
  {
    id: 'vision250d',
    name: 'Vision3 250D',
    family: 'Cinema',
    description: 'Motion-picture negative: low contrast, huge latitude',
    tone: { contrast: -0.15, highlights: -0.2, shadows: 0.15 },
    color: { saturation: -0.05, hslHue: hsl(0, 0, 0, -0.05) },
    curves: { rgb: p(0, 0.03, 0.5, 0.5, 1, 0.97) },
  },
  {
    id: 'vision500t',
    name: 'Vision3 500T',
    family: 'Cinema',
    description: 'Tungsten cinema stock: cool, soft, filmic night',
    tone: { contrast: -0.05, shadows: 0.1 },
    color: { temp: -0.15 },
    curves: { b: p(0, 0.04, 0.4, 0.42, 1, 0.98), r: p(0, 0, 0.7, 0.71, 1, 1) },
  },
  // ---- Other ----
  {
    id: 'lomo400',
    name: 'Lomo 400',
    family: 'Other',
    description: 'Saturated, contrasty, warm and a little wild',
    tone: { contrast: 0.15 },
    color: { temp: 0.08, saturation: 0.18, hslHue: hsl(0, 0, 0, 0, 0, 0.1) },
  },
  {
    id: 'vista200',
    name: 'Vista 200',
    family: 'Other',
    description: 'Strong reds against cool shadows',
    tone: { contrast: 0.05 },
    color: { tint: -0.02, saturation: 0.08, hslSat: hsl(0.2) },
    curves: { b: p(0, 0.03, 0.3, 0.31, 1, 1) },
  },
  // ---- Black & white ----
  { id: 'hp5', name: 'HP5 Plus', family: 'Black & white', bw: true, description: 'Classic, forgiving midtones', tone: { contrast: 0.05, shadows: 0.1 } },
  {
    id: 'trix',
    name: 'Tri-X 400',
    family: 'Black & white',
    bw: true,
    description: 'Gritty, punchy, deep blacks',
    tone: { contrast: 0.2, blacks: -0.1 },
    curves: { rgb: p(0, 0, 0.25, 0.2, 0.75, 0.8, 1, 1) },
  },
  { id: 'delta100', name: 'Delta 100', family: 'Black & white', bw: true, description: 'Smooth tones, gentle highlights', tone: { contrast: 0.05, highlights: -0.1 } },
  { id: 'panf', name: 'Pan F 50', family: 'Black & white', bw: true, description: 'Crisp, high contrast', tone: { contrast: 0.25, whites: 0.05 } },
  { id: 'tmax400', name: 'T-Max 400', family: 'Black & white', bw: true, description: 'Clean and modern', tone: { contrast: 0.1 } },
  { id: 'foma100', name: 'Fomapan 100', family: 'Black & white', bw: true, description: 'Vintage, lifted blacks', tone: { contrast: 0.12, blacks: 0.08 }, curves: { rgb: p(0, 0.04, 1, 1) } },
];

export function findLook(id: string | null | undefined): FilmLook | null {
  return (id && FILM_LOOKS.find((l) => l.id === id)) || null;
}
