const STORAGE_KEY = 'zen-mode-enabled';

function setZenMode(enabled) {
  document.documentElement.classList.toggle('zen-mode', enabled);
  localStorage.setItem(STORAGE_KEY, enabled ? '1' : '0');

  const button = document.getElementById('zen-mode-toggle');
  if (button) {
    button.setAttribute('aria-pressed', String(enabled));
    button.textContent = enabled ? '↙' : '🧐';
    button.title = enabled ? 'Exit Zen mode' : 'Enter Zen mode';
  }
}

function initZenMode() {
  const button = document.createElement('button');
  button.id = 'zen-mode-toggle';
  button.type = 'button';
  button.textContent = '🧐';
  button.title = 'Enter Zen mode';
  button.setAttribute('aria-label', 'Toggle Zen mode');
  button.setAttribute('aria-pressed', 'false');

  document.body.appendChild(button);

  const saved = localStorage.getItem(STORAGE_KEY) === '1';
  setZenMode(saved);

  button.addEventListener('click', () => {
    const enabled = !document.documentElement.classList.contains('zen-mode');
    setZenMode(enabled);
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initZenMode);
} else {
  initZenMode();
}