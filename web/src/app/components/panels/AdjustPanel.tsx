import type { PrintCurve } from '../../../engine/pipeline/recipe';
import { paramsFor } from '../../store/frames';
import { useCurrentFrame, useStore } from '../../store/rollStore';
import { Check, Section, Segmented, Slider, fmtStops } from '../controls';

export function AdjustPanel() {
  const frame = useCurrentFrame();
  const edit = useStore((s) => s.edit);
  const roll = useStore((s) => s.roll);
  if (!frame) return null;
  const autoEv = paramsFor(frame, roll, 'srgb')?.autoEv ?? 0;
  const t = frame.recipe.tone;
  const slide = (key: keyof Omit<typeof t, 'printCurve' | 'preserveHue'>, label: string, min = -1, max = 1, format?: (v: number) => string) => (
    <Slider label={label} value={t[key]} min={min} max={max} format={format} onChange={(v) => edit((d) => void (d.tone[key] = v), false)} />
  );
  return (
    <>
      <Section title="Print">
        <Segmented<PrintCurve>
          label="Paper"
          options={[
            { value: 'soft', label: 'Soft' },
            { value: 'standard', label: 'Standard' },
            { value: 'dense', label: 'Dense' },
            { value: 'linear', label: 'Linear', title: 'No paper curve: scene-linear output' },
          ]}
          value={t.printCurve}
          onChange={(v) => edit((d) => void (d.tone.printCurve = v))}
        />
        {slide('exposure', 'Exposure', -3, 3, fmtStops)}
        <Check
          label={`Auto exposure${frame.recipe.convert.autoExposure ? ` (${autoEv >= 0 ? '+' : ''}${autoEv.toFixed(2)} EV)` : ''}`}
          checked={frame.recipe.convert.autoExposure}
          onChange={(v) => edit((d) => void (d.convert.autoExposure = v))}
        />
        {slide('contrast', 'Contrast')}
      </Section>
      <Section title="Tone">
        {slide('highlights', 'Highlights')}
        {slide('shadows', 'Shadows')}
        {slide('whites', 'Whites')}
        {slide('blacks', 'Blacks')}
        <Check label="Preserve hue in highlights" checked={t.preserveHue} onChange={(v) => edit((d) => void (d.tone.preserveHue = v))} />
      </Section>
    </>
  );
}
