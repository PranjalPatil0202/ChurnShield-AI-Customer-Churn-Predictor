// ── Flash auto-dismiss ──────────────────────────────────────────
document.querySelectorAll('.flash').forEach(el => {
  setTimeout(() => {
    el.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
    el.style.opacity = '0';
    el.style.transform = 'translateX(40px)';
    setTimeout(() => el.remove(), 400);
  }, 4000);
});

// ── Toggle password visibility ──────────────────────────────────
function togglePw(id, btn) {
  const input = document.getElementById(id);
  const icon = btn.querySelector('i');
  if (input.type === 'password') {
    input.type = 'text';
    icon.classList.replace('fa-eye', 'fa-eye-slash');
  } else {
    input.type = 'password';
    icon.classList.replace('fa-eye-slash', 'fa-eye');
  }
}

// ── Form submit loading state ───────────────────────────────────
document.querySelectorAll('#loginForm, #regForm').forEach(form => {
  form.addEventListener('submit', function () {
    const btn = this.querySelector('button[type=submit]');
    if (!btn) return;
    const text = btn.querySelector('.btn-text');
    const loader = btn.querySelector('.btn-loader');
    if (text) text.classList.add('hidden');
    if (loader) loader.classList.remove('hidden');
    btn.disabled = true;
  });
});

// ── Animate numbers on dashboard ───────────────────────────────
function animateNumber(el, target, duration = 1000) {
  let start = 0;
  const step = target / (duration / 16);
  const timer = setInterval(() => {
    start += step;
    if (start >= target) { el.textContent = target; clearInterval(timer); return; }
    el.textContent = Math.floor(start);
  }, 16);
}

document.querySelectorAll('.stat-info .stat-num').forEach(el => {
  const val = parseInt(el.textContent);
  if (!isNaN(val) && val > 0) {
    el.textContent = '0';
    setTimeout(() => animateNumber(el, val), 300);
  }
});

// ── Smooth scroll for landing anchors ──────────────────────────
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    const target = document.querySelector(a.getAttribute('href'));
    if (target) { e.preventDefault(); target.scrollIntoView({ behavior: 'smooth' }); }
  });
});

// ── Navbar scroll effect ────────────────────────────────────────
const navbar = document.querySelector('.navbar');
if (navbar) {
  window.addEventListener('scroll', () => {
    navbar.style.borderBottomColor = window.scrollY > 40 ? 'var(--border2)' : 'var(--border)';
  });
}

// ── Staggered animation for feature cards ──────────────────────
const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry, i) => {
    if (entry.isIntersecting) {
      setTimeout(() => {
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
      }, i * 80);
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll('.feature-card').forEach((card, i) => {
  card.style.opacity = '0';
  card.style.transform = 'translateY(24px)';
  card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
  observer.observe(card);
});
