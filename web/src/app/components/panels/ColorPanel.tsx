import { useState } from 'react';
import { HSL_BANDS } from '../../../engine/pipeline/recipe';
import { useCurrentFrame, useStore } from '../../store/rollStore';
import { Section, Segmented, Slider } from '../controls';

const BAND_COLORS = ['#d9534f', '#e08a3c', '#d8c34a', '#5aa555', '#4fb3b0', '#4a74d0', '#8a5cc9', '#c95aa8'];
type HslKey = 'hslHue' | 'hslSat' | 'hslLum';

export function ColorPanel() {
  const frame = useCurrentFrame();
  const edit = useStore((s) => s.edit);
  const [mode, setMode] = useState<HslKey>('hslSat');
  if (!frame) return null;
  const c = frame.recipe.color;
  return (
    <>
      <Section title="White balance">
        <Slider label="Temperature" value={c.temp} min={-1} max={1} track="linear-gradient(to right, #5d8ad8, #8a8a8a, #e0a347)" onChange={(v) => edit((d) => void (d.color.temp = v), false)} />
        <Slider label="Tint" value={c.tint} min={-1} max={1} track="linear-gradient(to right, #5aa555, #8a8a8a, #c95aa8)" onChange={(v) => edit((d) => void (d.color.tint = v), false)} />
      </Section>
      <Section title="Presence">
        <Slider label="Vibrance" value={c.vibrance} min={-1} max={1} onChange={(v) => edit((d) => void (d.color.vibrance = v), false)} />
        <Slider label="Saturation" value={c.saturation} min={-1} max={1} onChange={(v) => edit((d) => void (d.color.saturation = v), false)} />
      </Section>
      <Section
        title="Colour mixer"
        action={
          <button
            type="button"
            className="btn small ghost"
            onClick={() =>
              edit((d) => {
                d.color[mode] = new Array(8).fill(0);
              })
            }
          >
            Reset
          </button>
        }
      >
        <Segmented<HslKey>
          options={[
            { value: 'hslHue', label: 'Hue' },
            { value: 'hslSat', label: 'Saturation' },
            { value: 'hslLum', label: 'Luminance' },
          ]}
          value={mode}
          onChange={setMode}
        />
        {HSL_BANDS.map((name, i) => (
          <Slider
            key={name}
            label={name}
            value={c[mode][i]}
            min={-1}
            max={1}
            track={`linear-gradient(to right, var(--line-strong), ${BAND_COLORS[i]})`}
            onChange={(v) => edit((d) => void (d.color[mode][i] = v), false)}
          />
        ))}
      </Section>
    </>
  );
}
