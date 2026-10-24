import type { PartialRecipe } from '../../engine/pipeline/recipe';

export interface BuiltinPreset {
  id: string;
  name: string;
  description: string;
  recipe: PartialRecipe;
}

/** Looks, not conversions: they leave film base and density measurement alone. */
export const BUILTIN_PRESETS: BuiltinPreset[] = [
  {
    id: 'neutral',
    name: 'Neutral',
    description: 'Standard print, no colour shifts',
    recipe: { tone: { printCurve: 'standard', contrast: 0, preserveHue: false }, color: { temp: 0, tint: 0, saturation: 0, vibrance: 0 } },
  },
  {
    id: 'soft',
    name: 'Soft Print',
    description: 'Low contrast, open shadows',
    recipe: { tone: { printCurve: 'soft', contrast: -0.1, shadows: 0.2 }, color: { saturation: -0.05 } },
  },
  {
    id: 'dense',
    name: 'Dense Print',
    description: 'Deeper blacks, punchier mids',
    recipe: { tone: { printCurve: 'dense', contrast: 0.1, blacks: -0.1 }, color: { vibrance: 0.1 } },
  },
  {
    id: 'warm',
    name: 'Warm Lab',
    description: 'A gentle golden print, like a warm-balanced minilab',
    recipe: { color: { temp: 0.12, tint: 0.03, vibrance: 0.08 } },
  },
  {
    id: 'cool',
    name: 'Cool Lab',
    description: 'Cleaner, cooler whites',
    recipe: { color: { temp: -0.1, tint: -0.02 } },
  },
  {
    id: 'faded',
    name: 'Faded',
    description: 'Lifted blacks, muted colour',
    recipe: { tone: { printCurve: 'soft', blacks: 0.35, whites: -0.1 }, color: { saturation: -0.2 } },
  },
];
