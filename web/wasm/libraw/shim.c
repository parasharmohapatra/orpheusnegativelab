/*
 * Orpheus Negative Lab — LibRaw WebAssembly shim.
 *
 * Decodes a RAW file held in memory to LINEAR camera RGB:
 *   no white balance (user_mul = 1), no gamma (1,1), no auto-brightness,
 *   no colour-space conversion (output_color = 0 / raw), no rotation.
 * Output is 16-bit interleaved RGB, black-subtracted and scaled so that
 * 65535 = sensor white (LibRaw's scale_colors with unit multipliers).
 *
 * The result is returned as a pointer to an orp_result struct with a fixed
 * layout (see the static asserts below; librawModule.ts reads the same
 * offsets). The LibRaw processor is closed before orp_decode returns; only the
 * pixel buffer stays alive until the caller passes the result to orp_free.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <emscripten/emscripten.h>

#include "libraw/libraw.h"

typedef struct orp_result {
  double timestamp;      /*   0  unix seconds, 0 if unknown */
  int32_t status;        /*   8  0 = ok, else LibRaw error code */
  int32_t width;         /*  12 */
  int32_t height;        /*  16 */
  uint32_t data_ptr;     /*  20  uint16 RGB, width*height*3 */
  int32_t flip;          /*  24  0,3,5,6 */
  uint32_t black;        /*  28 */
  uint32_t maximum;      /*  32 */
  float rgb_cam[9];      /*  36 */
  float cam_xyz[9];      /*  72 */
  float pre_mul[4];      /* 108 */
  float clip[3];         /* 124 */
  float iso;             /* 136 */
  float shutter;         /* 140 */
  float aperture;        /* 144 */
  float focal_len;       /* 148 */
  char make[64];         /* 152 */
  char model[64];        /* 216 */
  char error[256];       /* 280 */
  void *owner;           /* 536  private: block orp_free releases */
} orp_result;            /* 544 with tail padding */

_Static_assert(offsetof(orp_result, status) == 8, "layout");
_Static_assert(offsetof(orp_result, data_ptr) == 20, "layout");
_Static_assert(offsetof(orp_result, rgb_cam) == 36, "layout");
_Static_assert(offsetof(orp_result, cam_xyz) == 72, "layout");
_Static_assert(offsetof(orp_result, pre_mul) == 108, "layout");
_Static_assert(offsetof(orp_result, clip) == 124, "layout");
_Static_assert(offsetof(orp_result, iso) == 136, "layout");
_Static_assert(offsetof(orp_result, make) == 152, "layout");
_Static_assert(offsetof(orp_result, model) == 216, "layout");
_Static_assert(offsetof(orp_result, error) == 280, "layout");
_Static_assert(sizeof(orp_result) == 544, "layout"); /* padded to 8 */

/* Output values at or above this count as clipped (AHD can pull saturated
 * pixels a hair below 65535 at the edge of a blown region). */
#define ORP_CLIP_LEVEL 65500

static int normalize_flip(int flip) {
  switch ((flip + 3600) % 360) {
  case 270: return 5;
  case 180: return 3;
  case 90: return 6;
  }
  return flip;
}

static orp_result *fail(orp_result *r, libraw_data_t *lr, int code,
                        const char *where) {
  r->status = code ? code : -1;
  const char *msg = code ? libraw_strerror(code) : "unknown error";
  snprintf(r->error, sizeof r->error, "%s: %s", where, msg);
  if (lr) libraw_close(lr);
  return r;
}

EMSCRIPTEN_KEEPALIVE
const char *orp_version(void) { return libraw_version(); }

EMSCRIPTEN_KEEPALIVE
orp_result *orp_decode(const void *buf, size_t len, int half_size,
                       int user_qual) {
  orp_result *r = (orp_result *)calloc(1, sizeof(orp_result));
  if (!r) return NULL;

  libraw_data_t *lr = libraw_init(0);
  if (!lr) return fail(r, NULL, LIBRAW_UNSUFFICIENT_MEMORY, "libraw_init");

  libraw_output_params_t *p = &lr->params;
  p->output_bps = 16;
  p->gamm[0] = 1.0;
  p->gamm[1] = 1.0;
  p->no_auto_bright = 1;
  p->use_camera_wb = 0;
  p->use_auto_wb = 0;
  p->user_mul[0] = p->user_mul[1] = p->user_mul[2] = p->user_mul[3] = 1.0f;
  p->output_color = 0;
  p->highlight = 0;
  p->user_flip = 0;
  p->half_size = half_size ? 1 : 0;
  p->user_qual = user_qual;

  int rc = libraw_open_buffer(lr, buf, len);
  if (rc != LIBRAW_SUCCESS) return fail(r, lr, rc, "open");

  /* Orientation from the file, before user_flip overwrites it. */
  r->flip = normalize_flip(lr->sizes.flip);

  rc = libraw_unpack(lr);
  if (rc != LIBRAW_SUCCESS) return fail(r, lr, rc, "unpack");

  /* Colour metadata as the file describes it: dcraw_process overwrites
   * pre_mul with user_mul and zeroes black after subtracting it. */
  const libraw_colordata_t *col = &lr->color;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      r->rgb_cam[i * 3 + j] = col->rgb_cam[i][j];
      r->cam_xyz[i * 3 + j] = col->cam_xyz[i][j];
    }
  for (int c = 0; c < 4; c++) r->pre_mul[c] = col->pre_mul[c];
  if (r->pre_mul[3] == 0.f) r->pre_mul[3] = r->pre_mul[1];
  /* Per-channel black (cblack[0..3]) on top of the common level, averaged. */
  r->black = col->black + (col->cblack[0] + col->cblack[1] + col->cblack[2] +
                           col->cblack[3] + 2) / 4;
  r->maximum = col->maximum;

  rc = libraw_dcraw_process(lr);
  if (rc != LIBRAW_SUCCESS) return fail(r, lr, rc, "dcraw_process");

  int err = 0;
  libraw_processed_image_t *img = libraw_dcraw_make_mem_image(lr, &err);
  if (!img) return fail(r, lr, err, "make_mem_image");
  if (img->type != LIBRAW_IMAGE_BITMAP || img->bits != 16 ||
      (img->colors != 3 && img->colors != 1)) {
    libraw_dcraw_clear_mem(img);
    return fail(r, lr, LIBRAW_UNSUPPORTED_THUMBNAIL, "make_mem_image");
  }

  r->width = img->width;
  r->height = img->height;
  const size_t npix = (size_t)img->width * img->height;
  uint16_t *rgb;
  if (img->colors == 3) {
    /* libraw_processed_image_t is a header followed by the pixels in one
     * malloc block; hand the pixels out and free the block in orp_free. */
    rgb = (uint16_t *)img->data;
    r->owner = img;
  } else {
    /* Monochrome sensor: replicate into RGB. */
    rgb = (uint16_t *)malloc(npix * 3 * sizeof(uint16_t));
    if (!rgb) {
      libraw_dcraw_clear_mem(img);
      return fail(r, lr, LIBRAW_UNSUFFICIENT_MEMORY, "make_mem_image");
    }
    const uint16_t *m = (const uint16_t *)img->data;
    for (size_t i = 0; i < npix; i++)
      rgb[3 * i] = rgb[3 * i + 1] = rgb[3 * i + 2] = m[i];
    r->owner = rgb;
    libraw_dcraw_clear_mem(img);
  }

  size_t clipped[3] = {0, 0, 0};
  for (size_t i = 0; i < npix * 3; i += 3) {
    clipped[0] += rgb[i] >= ORP_CLIP_LEVEL;
    clipped[1] += rgb[i + 1] >= ORP_CLIP_LEVEL;
    clipped[2] += rgb[i + 2] >= ORP_CLIP_LEVEL;
  }

  r->data_ptr = (uint32_t)(uintptr_t)rgb;
  for (int c = 0; c < 3; c++)
    r->clip[c] = npix ? (float)clipped[c] / (float)npix : 0.f;

  r->iso = lr->other.iso_speed;
  r->shutter = lr->other.shutter;
  r->aperture = lr->other.aperture;
  r->focal_len = lr->other.focal_len;
  r->timestamp = (double)lr->other.timestamp;
  strncpy(r->make, lr->idata.make, sizeof r->make - 1);
  strncpy(r->model, lr->idata.model, sizeof r->model - 1);

  libraw_close(lr);
  return r;
}

EMSCRIPTEN_KEEPALIVE
void orp_free(orp_result *r) {
  if (!r) return;
  free(r->owner);
  free(r);
}
