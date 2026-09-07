import { useEffect, useId, useRef } from 'react';

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/**
 * Accessible modal shell: labels the dialog, traps Tab inside it, closes on
 * Escape, and restores focus to whatever opened it.
 * Pass onClose={null} for dialogs that require an explicit choice.
 */
export default function Modal({
  title,
  onClose,
  children,
  panelClassName = '',
  overlayClassName = 'bg-black/55',
  zIndexClassName = 'z-[90]',
  labelledBy,
}) {
  const panelRef = useRef(null);
  const previouslyFocused = useRef(null);
  const generatedId = useId();
  const titleId = labelledBy ?? generatedId;

  useEffect(() => {
    previouslyFocused.current = document.activeElement;
    const panel = panelRef.current;
    const first = panel?.querySelector(FOCUSABLE);
    (first ?? panel)?.focus();

    const onKeyDown = (event) => {
      if (event.key === 'Escape' && onClose) {
        event.stopPropagation();
        onClose();
        return;
      }
      if (event.key !== 'Tab' || !panel) return;

      const items = Array.from(panel.querySelectorAll(FOCUSABLE)).filter(
        (el) => el.offsetParent !== null || el === document.activeElement
      );
      if (!items.length) {
        event.preventDefault();
        return;
      }
      const firstItem = items[0];
      const lastItem = items[items.length - 1];
      if (event.shiftKey && document.activeElement === firstItem) {
        event.preventDefault();
        lastItem.focus();
      } else if (!event.shiftKey && document.activeElement === lastItem) {
        event.preventDefault();
        firstItem.focus();
      }
    };

    document.addEventListener('keydown', onKeyDown, true);
    return () => {
      document.removeEventListener('keydown', onKeyDown, true);
      previouslyFocused.current?.focus?.();
    };
  }, [onClose]);

  return (
    <div className={`fixed inset-0 ${zIndexClassName} flex items-center justify-center p-4 ${overlayClassName}`}>
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        className={panelClassName}
      >
        {title ? <h2 id={titleId} className="text-xl font-black tracking-tight">{title}</h2> : null}
        {children}
      </div>
    </div>
  );
}
