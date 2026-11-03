export {}

declare module 'react' {
  // Lets slides pass CSS custom properties (--v, --tone) through the style prop.
  interface CSSProperties {
    [property: `--${string}`]: string | number | undefined
  }
}
