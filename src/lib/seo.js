import { useEffect } from 'react';

const SITE_NAME = 'SignCards';
const SITE_URL = 'https://signcards.app';

function setMeta(name, content, attr = 'name') {
  let tag = document.head.querySelector(`meta[${attr}="${name}"]`);
  if (!tag) {
    tag = document.createElement('meta');
    tag.setAttribute(attr, name);
    document.head.appendChild(tag);
  }
  tag.setAttribute('content', content);
}

function setCanonical(path) {
  let link = document.head.querySelector('link[rel="canonical"]');
  if (!link) {
    link = document.createElement('link');
    link.setAttribute('rel', 'canonical');
    document.head.appendChild(link);
  }
  link.setAttribute('href', `${SITE_URL}${path}`);
}

/**
 * Updates document title, meta description, and canonical URL per route.
 * This is a client-side stopgap: it only takes effect after JavaScript runs,
 * so it helps modern crawlers (Google) more than ones that don't execute JS.
 */
export function useDocumentMeta({ title, description, path, noindex = false }) {
  useEffect(() => {
    const fullTitle = title ? `${title} | ${SITE_NAME}` : SITE_NAME;
    document.title = fullTitle;
    if (description) {
      setMeta('description', description);
      setMeta('og:description', description, 'property');
      setMeta('twitter:description', description);
    }
    setMeta('og:title', fullTitle, 'property');
    setMeta('twitter:title', fullTitle);
    setMeta('robots', noindex ? 'noindex, follow' : 'index, follow');
    if (path) {
      setCanonical(path);
      setMeta('og:url', `${SITE_URL}${path}`, 'property');
    }
  }, [title, description, path, noindex]);
}
