import { beforeEach, describe, expect, it } from 'vitest';
import { newFrameEntry, useStore } from '../../src/app/store/rollStore';

const file = (name: string) => new File([new Uint8Array(4)], name);

function setup() {
  useStore.setState({ frames: [], currentId: null, selected: [], anchorId: null, editSnapshots: null, lastGroup: null, redoGroup: null, autoSync: true });
  const entries = ['a.cr2', 'b.cr2', 'c.cr2', 'd.cr2'].map((n) => newFrameEntry(file(n), 'raw', n));
  useStore.getState().addFrames(entries);
  return useStore.getState().frames.map((f) => f.id);
}
const recipe = (id: string) => useStore.getState().frames.find((f) => f.id === id)!.recipe;

describe('bulk editing', () => {
  let ids: string[];
  beforeEach(() => {
    ids = setup();
  });

  it('shift-click selects a range from the anchor and keeps the clicked frame current', () => {
    const s = useStore.getState();
    s.setCurrent(ids[0]);
    s.toggleSelect(ids[2], 'range');
    expect(useStore.getState().selected).toEqual(ids.slice(0, 3));
    expect(useStore.getState().currentId).toBe(ids[2]);
  });

  it('applies slider edits to every selected frame and undoes them as one step', () => {
    const s = useStore.getState();
    s.setCurrent(ids[0]);
    s.toggleSelect(ids[1], 'toggle');
    s.edit((r) => void (r.tone.exposure = 0.5), false);
    s.edit((r) => void (r.tone.exposure = 0.8), false);
    s.commit();
    expect(recipe(ids[0]).tone.exposure).toBe(0.8);
    expect(recipe(ids[1]).tone.exposure).toBe(0.8);
    expect(recipe(ids[2]).tone.exposure).toBe(0);
    useStore.getState().undo();
    expect(recipe(ids[0]).tone.exposure).toBe(0);
    expect(recipe(ids[1]).tone.exposure).toBe(0);
    useStore.getState().redo();
    expect(recipe(ids[1]).tone.exposure).toBe(0.8);
  });

  it('keeps local edits (crop) on the current frame only', () => {
    const s = useStore.getState();
    s.setCurrent(ids[0]);
    s.toggleSelect(ids[1], 'toggle');
    s.edit((r) => void (r.geometry.crop = { x: 0.1, y: 0.1, w: 0.5, h: 0.5 }), true, true);
    expect(recipe(ids[1]).geometry.crop.w).toBe(1);
    expect(recipe(useStore.getState().currentId!).geometry.crop.w).toBe(0.5);
  });

  it('edits only the current frame when auto-sync is off', () => {
    const s = useStore.getState();
    s.setCurrent(ids[0]);
    s.toggleSelect(ids[1], 'toggle');
    useStore.setState({ autoSync: false });
    const cur = useStore.getState().currentId!;
    useStore.getState().edit((r) => void (r.color.saturation = 0.3));
    const other = ids.slice(0, 2).find((id) => id !== cur)!;
    expect(recipe(cur).color.saturation).toBe(0.3);
    expect(recipe(other).color.saturation).toBe(0);
  });

  it('bulk actions apply to the selection with one undo', () => {
    const s = useStore.getState();
    s.editFrames([ids[1], ids[3]], (r) => void (r.film.type = 'bw'));
    expect(recipe(ids[1]).film.type).toBe('bw');
    expect(recipe(ids[3]).film.type).toBe('bw');
    useStore.getState().setCurrent(ids[1]);
    useStore.setState({ lastGroup: [ids[1], ids[3]] });
    useStore.getState().undo();
    expect(recipe(ids[3]).film.type).toBe('c41');
  });
});

describe('selection clicks', () => {
  it('a plain click inside a multi-selection keeps the selection', () => {
    const ids = setup();
    const s = useStore.getState();
    s.setCurrent(ids[0]);
    s.toggleSelect(ids[2], 'range');
    useStore.getState().toggleSelect(ids[1], 'single');
    expect(useStore.getState().selected).toHaveLength(3);
    expect(useStore.getState().currentId).toBe(ids[1]);
    useStore.getState().toggleSelect(ids[3], 'single');
    expect(useStore.getState().selected).toEqual([ids[3]]);
  });
});
