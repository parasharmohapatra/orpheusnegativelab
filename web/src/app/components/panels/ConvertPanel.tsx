import { defaultRecipe, setFilmType, type BalanceMode, type FilmType, type RangeMode } from '../../../engine/pipeline/recipe';
import { resolveEndpoints } from '../../../engine/pipeline/params';
import { srgbEncode } from '../../../engine/color/transfer';
import { useCurrentFrame, useStore } from '../../store/rollStore';
import { Check, Section, Segmented, Slider } from '../controls';
import { IconPipette } from '../icons';

const toCss = (rgb: number[]) => {
  const m = Math.max(...rgb, 1e-6);
  // Show the base as it looks on the light table: normalized so it's visible.
  const v = rgb.map((c) => Math.round(srgbEncode((c / m) * 0.9) * 255));
  return `rgb(${v.join(',')})`;
};

export function ConvertPanel() {
  const frame = useCurrentFrame();
  const roll = useStore((s) => s.roll);
  const edit = useStore((s) => s.edit);
  const picker = useStore((s) => s.picker);
  const set = useStore((s) => s.set);
  if (!frame) return null;
  const r = frame.recipe;
  const resolved = frame.stats ? resolveEndpoints(r, frame.stats, roll) : null;
  const base = resolved?.base;
  const lowPrecision = frame.meta?.lowPrecision;

  return (
    <>
      <Section title="Film">
        <Segmented<FilmType>
          options={[
            { value: 'c41', label: 'Colour neg', title: 'C-41 / ECN-2 colour negative' },
            { value: 'bw', label: 'B&W neg' },
            { value: 'e6', label: 'Slide', title: 'E-6 positive transparency' },
            { value: 'photo', label: 'Photo', title: 'An ordinary photo: no inversion, just film looks and adjustments' },
          ]}
          value={r.film.type}
          onChange={(v) => edit((d) => setFilmType(d, v))}
        />
        {r.film.type === 'photo' && (
          <p className="hint">
            An ordinary photo: nothing is inverted or measured. Pick a film look under <em>Looks</em>, then fine-tune in Adjust, Colour and Curves.
          </p>
        )}
        {lowPrecision && r.film.type !== 'photo' && <p className="hint warn">This is an 8-bit file. Inverting stretches it a lot, so expect banding; use a RAW or 16-bit TIFF scan for best results.</p>}
      </Section>

      {r.film.type !== 'photo' && (
      <>
      <Section title="Film base">
        <div className="swatch-row">
          <div className="swatch" style={{ background: base ? toCss(base) : 'transparent' }} title="Measured film base (orange mask)" />
          <div style={{ flex: 1, fontSize: 12.5, color: 'var(--text-muted)', lineHeight: 1.45 }}>
            {r.film.baseMode === 'manual'
              ? 'Sampled by hand'
              : frame.stats?.baseFromBorder
                ? roll && roll.frames > 1
                  ? `From the rebate, median of ${roll.frames} frames`
                  : 'Measured on the film rebate'
                : 'Estimated from the thinnest density. Sample the rebate for accuracy.'}
          </div>
        </div>
        <div className="row">
          <button type="button" className={`btn small${picker === 'base' ? ' active' : ''}`} onClick={() => set('picker', picker === 'base' ? null : 'base')}>
            <IconPipette /> Sample base
          </button>
          <button
            type="button"
            className="btn small"
            disabled={r.film.baseMode === 'auto'}
            onClick={() =>
              edit((d) => {
                d.film.baseMode = 'auto';
                d.film.baseRgb = null;
              })
            }
          >
            Auto
          </button>
        </div>
      </Section>

      <Section title="Colour balance">
        <Segmented<RangeMode>
          label="Measure per"
          options={[
            { value: 'frame', label: 'Frame', title: 'Balance every frame on its own (like NLP auto)' },
            { value: 'roll', label: 'Roll', title: 'Share the film response across the roll; keep exposure per frame' },
            { value: 'blend', label: 'Blend', title: 'Choose how much each frame may differ from the roll' },
          ]}
          value={r.convert.rangeMode}
          onChange={(v) => edit((d) => void (d.convert.rangeMode = v))}
        />
        {r.convert.rangeMode === 'blend' && (
          <>
            <Slider label="Frame exposure" value={r.convert.lambdaExp} min={0} max={1} reset={0.8} format={(v) => `${Math.round(v * 100)}%`} onChange={(v) => edit((d) => void (d.convert.lambdaExp = v), false)} />
            <Slider label="Frame colour" value={r.convert.lambdaCol} min={0} max={1} reset={0.2} format={(v) => `${Math.round(v * 100)}%`} onChange={(v) => edit((d) => void (d.convert.lambdaCol = v), false)} />
          </>
        )}
        <Segmented<BalanceMode>
          label="Balance"
          options={[
            { value: 'endpoints+mid', label: 'Auto', title: 'Neutral blacks, whites and near-grey midtones' },
            { value: 'endpoints', label: 'Ends', title: 'Neutral blacks and whites only' },
            { value: 'none', label: 'Off', title: 'Only remove the orange mask' },
          ]}
          value={r.convert.balance === 'neutral' ? 'endpoints' : r.convert.balance}
          onChange={(v) => edit((d) => void (d.convert.balance = v))}
        />
        {r.convert.balance === 'endpoints+mid' && (
          <Slider label="Midtone balance" value={r.convert.midStrength} min={0} max={1} reset={0.75} format={(v) => `${Math.round(v * 100)}%`} onChange={(v) => edit((d) => void (d.convert.midStrength = v), false)} />
        )}
        <div className="row">
          <button type="button" className={`btn small${picker === 'neutral' ? ' active' : ''}`} onClick={() => set('picker', picker === 'neutral' ? null : 'neutral')}>
            <IconPipette /> Pick neutral
          </button>
          <button type="button" className="btn small" disabled={r.convert.neutralOffset.every((v) => v === 0)} onClick={() => edit((d) => void (d.convert.neutralOffset = [0, 0, 0]))}>
            Clear
          </button>
        </div>
      </Section>

      <Section title="Density range" defaultOpen={false}>
        <Segmented<'fill' | 'physical'>
          label="Auto range"
          options={[
            { value: 'fill', label: 'Fill print', title: 'Stretch each frame to the full print range, like a lab scanner' },
            { value: 'physical', label: 'Scene contrast', title: "Keep the scene's own contrast (flat light stays flat)" },
          ]}
          value={r.convert.rangeAuto}
          onChange={(v) =>
            edit((d) => {
              d.convert.rangeAuto = v;
              d.convert.sceneStops = null;
            })
          }
        />
        <Slider
          label="Scene range"
          value={r.convert.sceneStops ?? resolved?.ends.sceneStops ?? 8.5}
          min={4}
          max={12}
          step={0.05}
          reset={resolved?.ends.sceneStops ?? 8.5}
          format={(v) => `${v.toFixed(1)} stops${r.convert.sceneStops === null ? ' · auto' : ''}`}
          onChange={(v) => edit((d) => void (d.convert.sceneStops = v), false)}
        />
        <Slider label="Black clip" value={r.convert.blackClip} min={0} max={3} step={0.05} reset={defaultRecipe().convert.blackClip} format={(v) => `${v.toFixed(2)}%`} onChange={(v) => edit((d) => void (d.convert.blackClip = v), false)} />
        <Slider label="White clip" value={r.convert.whiteClip} min={0} max={3} step={0.05} reset={defaultRecipe().convert.whiteClip} format={(v) => `${v.toFixed(2)}%`} onChange={(v) => edit((d) => void (d.convert.whiteClip = v), false)} />
        <div className="row">
          <button type="button" className="btn small" disabled={r.convert.sceneStops === null} onClick={() => edit((d) => void (d.convert.sceneStops = null))}>
            Auto range
          </button>
        </div>
        <Check label="Ignore film border when measuring" checked={r.geometry.autoFrame} onChange={(v) => edit((d) => void (d.geometry.autoFrame = v))} />
        <p className="hint">
          Lower <em>scene range</em> for punchier conversions, raise it to hold more highlight and shadow detail.
        </p>
      </Section>

      </>
      )}

      {r.film.type === 'bw' && (
        <Section title="B&W channel mix">
          {(['Red', 'Green', 'Blue'] as const).map((name, c) => (
            <Slider
              key={name}
              label={name}
              value={r.film.bwMix[c]}
              min={0}
              max={1}
              reset={defaultRecipe().film.bwMix[c]}
              format={(v) => `${Math.round(v * 100)}`}
              onChange={(v) => edit((d) => void (d.film.bwMix[c] = v), false)}
            />
          ))}
        </Section>
      )}
    </>
  );
}
