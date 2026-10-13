import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

/**
 * Joins class names and resolves conflicting Tailwind utilities, keeping the last one.
 * @param inputs - class values, including conditional objects and arrays
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
