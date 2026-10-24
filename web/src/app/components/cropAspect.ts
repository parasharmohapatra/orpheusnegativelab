/** Locked crop aspect (width / height in pixels), or null for free. Shared by the crop panel and overlay. */
let ratio: number | null = null;
export const cropAspect = {
  get: () => ratio,
  set: (r: number | null) => {
    ratio = r;
  },
};
