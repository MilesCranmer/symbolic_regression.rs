import { useEffect, useState } from "react";

export function formatSci(x: number): string {
  if (!Number.isFinite(x)) return String(x);
  return x.toExponential(3);
}

export function usePrefersDark(): boolean {
  const get = () => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return false;
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  };

  const [dark, setDark] = useState(get);

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return;
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => setDark(media.matches);
    // Safari supports addListener/removeListener, modern browsers support addEventListener.
    if (typeof media.addEventListener === "function") media.addEventListener("change", onChange);
    else (media as any).addListener?.(onChange);
    return () => {
      if (typeof media.removeEventListener === "function") media.removeEventListener("change", onChange);
      else (media as any).removeListener?.(onChange);
    };
  }, []);

  return dark;
}

export function plotLayoutBase(prefersDark: boolean) {
  const axisBase = prefersDark
    ? {
        gridcolor: "rgba(127,127,127,0.25)",
        zerolinecolor: "rgba(127,127,127,0.25)",
        tickcolor: "rgba(235,235,235,0.6)",
        linecolor: "rgba(235,235,235,0.6)"
      }
    : {
        gridcolor: "rgba(0,0,0,0.12)",
        zerolinecolor: "rgba(0,0,0,0.18)",
        tickcolor: "rgba(0,0,0,0.45)",
        linecolor: "rgba(0,0,0,0.45)"
      };

  return {
    ...(prefersDark
      ? {
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { color: "rgba(235,235,235,0.95)" }
        }
      : {}),
    xaxis: {
      ...axisBase,
      showline: true,
      linewidth: 1,
      mirror: true,
      ticks: "outside",
      showgrid: true,
      zeroline: true
    },
    yaxis: {
      ...axisBase,
      showline: true,
      linewidth: 1,
      mirror: true,
      ticks: "outside",
      showgrid: true,
      zeroline: true
    }
  };
}

export function sortXY(x: number[], y: number[]): { x: number[]; y: number[] } {
  const pairs = x.map((xi, i) => ({ x: xi, y: y[i] ?? NaN })).filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
  pairs.sort((a, b) => a.x - b.x);
  return { x: pairs.map((p) => p.x), y: pairs.map((p) => p.y) };
}

export function copyToClipboard(text: string): void {
  void navigator.clipboard.writeText(text);
}
