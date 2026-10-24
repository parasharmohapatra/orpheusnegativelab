import { defaultRecipe, pickSections, type RecipeSection } from '../../engine/pipeline/recipe';
import { useCurrentFrame, useStore, type Tool } from '../store/rollStore';
import { Histogram } from './Histogram';
import { AdjustPanel } from './panels/AdjustPanel';
import { ColorPanel } from './panels/ColorPanel';
import { ConvertPanel } from './panels/ConvertPanel';
import { CropPanel } from './panels/CropPanel';
import { CurvesPanel } from './panels/CurvesPanel';
import { PresetsPanel } from './panels/PresetsPanel';

const TITLES: Record<Tool, string> = {
  convert: 'Convert',
  adjust: 'Adjust',
  color: 'Colour',
  curves: 'Curves',
  crop: 'Crop',
  presets: 'Looks',
};

/** Which recipe sections "Reset" clears on each tool. */
const RESETS: Partial<Record<Tool, RecipeSection[]>> = {
  convert: ['film', 'convert'],
  adjust: ['tone'],
  color: ['color'],
  curves: ['curves'],
  crop: ['geometry'],
  presets: ['look'],
};

export function Panel() {
  const tool = useStore((s) => s.tool);
  const frame = useCurrentFrame();
  const selected = useStore((s) => s.selected);
  const set = useStore((s) => s.set);
  const applyToFrames = useStore((s) => s.applyToFrames);
  const resetSections = RESETS[tool];
  const autoSync = useStore((s) => s.autoSync);
  const bulk = autoSync && selected.length > 1;

  return (
    <aside className="panel" aria-label="Adjustments">
      <div className="panel-scroll">
        <h2 className="panel-title">
          {TITLES[tool]}
          {frame && frame.history.length > 0 && <small>{frame.history.length} edits</small>}
        </h2>
        {bulk && tool !== 'crop' && tool !== 'presets' && (
          <p className="bulk-note">
            Adjusting <b>{selected.length} frames</b> together. Turn off “Edit all” below the photo to edit just this one.
          </p>
        )}
        {bulk && tool === 'crop' && <p className="bulk-note">Crop and straighten apply to this frame only. Rotation applies to all {selected.length} selected.</p>}
        {frame && tool !== 'presets' && tool !== 'crop' && <Histogram />}
        {!frame && <p className="hint">Open a roll to begin.</p>}
        {tool === 'convert' && <ConvertPanel />}
        {tool === 'adjust' && <AdjustPanel />}
        {tool === 'color' && <ColorPanel />}
        {tool === 'curves' && <CurvesPanel />}
        {tool === 'crop' && <CropPanel />}
        {tool === 'presets' && <PresetsPanel />}
      </div>
      {frame && (
        <div className="panel-foot">
          {resetSections && (
            <button type="button" className="btn small block" onClick={() => applyToFrames(bulk ? selected : [frame.id], pickSections(defaultRecipe(), resetSections))}>
              Reset {TITLES[tool].toLowerCase()}{bulk ? ` (${selected.length})` : ''}
            </button>
          )}
          <button type="button" className="btn small block" disabled={selected.length < 2} onClick={() => set('syncOpen', true)} title="Copy settings from this frame to the selected frames">
            Sync{selected.length > 1 ? ` (${selected.length})` : ''}
          </button>
        </div>
      )}
    </aside>
  );
}
