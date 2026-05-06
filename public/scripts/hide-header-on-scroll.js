let lastY = window.scrollY;
const threshold = 5;

function onScroll() {
  const currentY = window.scrollY;
  const scrollingDown = currentY > lastY + threshold;
  const scrollingUp = currentY < lastY - threshold;

  if (scrollingDown && currentY > 80) {
    document.documentElement.classList.add('header-hidden');
  }

  if (scrollingUp || currentY < 80) {
    document.documentElement.classList.remove('header-hidden');
  }

  lastY = currentY;
}

window.addEventListener('scroll', onScroll, { passive: true });