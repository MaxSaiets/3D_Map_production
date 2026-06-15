// Learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom'

// next-intl: компоненти тепер скрізь викликають useTranslations — у unit-тестах
// немає NextIntlClientProvider, тож мокаємо як passthrough (повертає ключ).
jest.mock('next-intl', () => {
  // Резолвимо РЕАЛЬНІ укр-переклади з messages/uk.json — так компоненти рендерять
  // справжній текст, і тести (що перевіряють укр-рядки) проходять.
  const messages = require('./messages/uk.json');
  const get = (path) => path.split('.').reduce((o, k) => (o == null ? undefined : o[k]), messages);
  const makeT = (ns) => {
    const t = (key, vals) => {
      const v = get(ns ? `${ns}.${key}` : key);
      if (typeof v !== 'string') return key;
      return vals ? v.replace(/\{(\w+)\}/g, (_, k) => (vals[k] != null ? String(vals[k]) : `{${k}}`)) : v;
    };
    t.rich = (key) => t(key);
    t.raw = (key) => get(ns ? `${ns}.${key}` : key) ?? key;
    t.markup = (key) => t(key);
    t.has = (key) => get(ns ? `${ns}.${key}` : key) != null;
    return t;
  };
  return {
    useTranslations: (ns) => makeT(ns),
    useLocale: () => 'uk',
    useFormatter: () => ({ number: (n) => String(n), dateTime: (d) => String(d) }),
    useMessages: () => messages,
    NextIntlClientProvider: ({ children }) => children,
  };
});

// @/i18n/navigation — обгортки next-intl над next/navigation; мокаємо на прості версії.
jest.mock('@/i18n/navigation', () => ({
  Link: ({ children, href, ...rest }) => <a href={typeof href === 'string' ? href : '#'} {...rest}>{children}</a>,
  useRouter: () => ({ push: jest.fn(), replace: jest.fn(), prefetch: jest.fn(), back: jest.fn() }),
  usePathname: () => '/',
  redirect: jest.fn(),
  getPathname: () => '/',
}));

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(), // deprecated
    removeListener: jest.fn(), // deprecated
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
})

// Mock Leaflet
jest.mock('leaflet', () => ({
  map: jest.fn(),
  tileLayer: jest.fn(),
  icon: jest.fn(),
  Marker: jest.fn(),
  Control: {
    Draw: jest.fn(),
  },
  Draw: {
    Event: {
      CREATED: 'draw:created',
      EDITED: 'draw:edited',
      DELETED: 'draw:deleted',
    },
  },
}))

// Mock react-leaflet
jest.mock('react-leaflet', () => ({
  MapContainer: ({ children }) => <div data-testid="map-container">{children}</div>,
  TileLayer: () => <div data-testid="tile-layer" />,
  useMap: () => ({
    on: jest.fn(),
    off: jest.fn(),
    addLayer: jest.fn(),
    addControl: jest.fn(),
    removeControl: jest.fn(),
  }),
}))

