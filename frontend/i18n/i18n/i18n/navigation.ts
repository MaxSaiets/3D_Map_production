import { createNavigation } from "next-intl/navigation";
import { routing } from "./routing";

// Locale-aware Link / useRouter / usePathname / redirect.
export const { Link, redirect, usePathname, useRouter, getPathname } =
  createNavigation(routing);
