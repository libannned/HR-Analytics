import type { ReactNode } from "react";
import { headers } from "next/headers";
import "./globals.css";

const title = "Greenlight — Content approvals, all in one place";
const description = "Review and approve creator content across TikTok Shop, Instagram Reels, LTK, and ShopMy.";

export async function generateMetadata() {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host") ?? "localhost:3000";
  const protocol = requestHeaders.get("x-forwarded-proto") ?? (host.startsWith("localhost") ? "http" : "https");
  const image = `${protocol}://${host}/og.png`;
  return {
    title,
    description,
    openGraph: { title, description, images: [{ url: image, width: 1732, height: 909, alt: "Greenlight creator content approval workspace" }] },
    twitter: { card: "summary_large_image", title, description, images: [image] },
  };
}

export default function RootLayout({ children }: { children: ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}
