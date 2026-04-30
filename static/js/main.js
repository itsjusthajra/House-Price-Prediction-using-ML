// Global Plotly config for consistent dark theme
const PLOTLY_CONFIG = {
  responsive: true,
  displayModeBar: true,
  modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
  displaylogo: false,
};

const PLOTLY_LAYOUT_PATCH = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { color: '#38404a', family: 'Inter, sans-serif' },
  margin: { l: 48, r: 16, t: 48, b: 40 },
};

function renderChart(divId, chartJson) {
  const el = document.getElementById(divId);
  if (!el || !chartJson) return;

  try {
    const data = typeof chartJson === 'string' ? JSON.parse(chartJson) : chartJson;
    const layout = Object.assign({}, data.layout, PLOTLY_LAYOUT_PATCH);
    Plotly.newPlot(el, data.data, layout, PLOTLY_CONFIG);
  } catch (e) {
    console.error(`Failed to render chart #${divId}:`, e);
  }
}

// Animate numbers counting up
function animateNumber(el, target, duration = 1200, prefix = '', suffix = '') {
  const start = performance.now();
  const from = parseFloat(el.dataset.from || 0);

  function update(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = from + (target - from) * eased;
    el.textContent = prefix + current.toLocaleString('en-IN', { maximumFractionDigits: 2 }) + suffix;
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

// Smooth scroll helper
function scrollTo(id) {
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Toast notifications
function showToast(message, type = 'info') {
  const colors = { info: '#6C63FF', success: '#43E97B', error: '#FF6584' };
  const toast = document.createElement('div');
  toast.style.cssText = `
    position: fixed; bottom: 1.5rem; right: 1.5rem; z-index: 9999;
    background: #13132a; border: 1px solid ${colors[type]}55;
    color: #e2e8f0; padding: 0.8rem 1.2rem; border-radius: 12px;
    font-size: 0.9rem; box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    display: flex; align-items: center; gap: 0.6rem;
    transform: translateX(120%); transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
  `;
  toast.innerHTML = `<span style="color:${colors[type]};font-size:1.1rem">${type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ'}</span>${message}`;
  document.body.appendChild(toast);
  requestAnimationFrame(() => { toast.style.transform = 'translateX(0)'; });
  setTimeout(() => {
    toast.style.transform = 'translateX(120%)';
    setTimeout(() => toast.remove(), 400);
  }, 3500);
}

// Debounce utility
function debounce(fn, wait) {
  let timer;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), wait); };
}

// Resize charts on window resize
window.addEventListener('resize', debounce(() => {
  document.querySelectorAll('.plotly-chart').forEach(el => {
    if (el._fullLayout) Plotly.relayout(el, {});
  });
}, 300));

// Animate stat cards when they come into view
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.opacity = '1';
      entry.target.style.transform = 'translateY(0)';
    }
  });
}, { threshold: 0.1 });

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.stat-card, .card-glass').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(16px)';
    el.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
    observer.observe(el);
  });
});
