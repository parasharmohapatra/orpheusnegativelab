import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import '@fontsource/cormorant-garamond/500.css';
import '@fontsource/cormorant-garamond/500-italic.css';
import '@fontsource-variable/inter';
import './app/theme/app.css';
import { App } from './app/App';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

if (import.meta.env.DEV) {
  // ?devload=/@fs/abs/path/a.CR2,/@fs/abs/path/b.CR2 opens files served by the dev server.
  void import('./app/store/rollStore').then((m) => ((window as unknown as { __onl: unknown }).__onl = m));
  const q = new URLSearchParams(location.search);
  const asPhotos = q.has('devphotos');
  const list = q.get('devload') ?? q.get('devphotos');
  if (list) {
    void import('./app/store/frames').then(async ({ openFiles }) => {
      const files = await Promise.all(
        list.split(',').map(async (p) => {
          const blob = await (await fetch(p)).blob();
          return new File([blob], decodeURIComponent(p.split('/').pop()!));
        }),
      );
      await openFiles(files, { asPhotos });
    });
  }
}
