'use strict';

const form         = document.getElementById('listing-form');
const submitBtn    = document.getElementById('submit-btn');
const btnText      = document.getElementById('btn-text');
const btnSpinner   = document.getElementById('btn-spinner');
const resultPanel  = document.getElementById('result-panel');
const errorBanner  = document.getElementById('error-banner');

// ── Populate neighbourhood dropdown ──────────────────────────
async function loadNeighbourhoods() {
  const select = document.getElementById('neighbourhood');
  try {
    const res  = await fetch('/neighbourhoods');
    const data = await res.json();
    select.innerHTML = '';
    if (!data.neighbourhoods || data.neighbourhoods.length === 0) {
      select.innerHTML = '<option value="Le Plateau-Mont-Royal">Le Plateau-Mont-Royal</option>';
      return;
    }
    data.neighbourhoods.forEach(n => {
      const opt = document.createElement('option');
      opt.value = n;
      opt.textContent = n;
      if (n === 'Le Plateau-Mont-Royal') opt.selected = true;
      select.appendChild(opt);
    });
  } catch {
    select.innerHTML = '<option value="Le Plateau-Mont-Royal">Le Plateau-Mont-Royal</option>';
  }
}

// ── Form submission ───────────────────────────────────────────
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  setLoading(true);
  clearResults();

  const payload = buildPayload();

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      showError(err.detail || 'Prediction failed.');
      return;
    }

    const data = await res.json();
    renderResult(data);
  } catch (err) {
    showError('Could not reach the server. Is it running?');
  } finally {
    setLoading(false);
  }
});

// ── Payload builder ───────────────────────────────────────────
function buildPayload() {
  const fd = new FormData(form);
  const p  = Object.fromEntries(fd.entries());

  return {
    room_type:                   p.room_type,
    neighbourhood:               p.neighbourhood,
    accommodates:                parseInt(p.accommodates, 10),
    bathrooms:                   parseFloat(p.bathrooms),
    bedrooms:                    parseInt(p.bedrooms, 10),
    beds:                        parseInt(p.beds, 10),
    property_type:               p.property_type || 'Apartment',
    instant_bookable:            document.getElementById('instant_bookable').checked,
    host_total_listings_count:   parseInt(p.host_total_listings_count, 10),
    latitude:                    parseFloat(p.latitude),
    longitude:                   parseFloat(p.longitude),
    minimum_nights:              parseInt(p.minimum_nights, 10),
    availability_365:            parseInt(p.availability_365, 10),
    number_of_reviews:           parseInt(p.number_of_reviews, 10),
    season:                      p.season,
    description:                 p.description || '',
    has_valid_image:             document.getElementById('has_valid_image').checked,
  };
}

// ── Result rendering ──────────────────────────────────────────
function renderResult(data) {
  document.getElementById('result-amount').textContent = fmt(data.predicted_price_cad);
  document.getElementById('result-low').textContent    = fmt(data.price_range.low);
  document.getElementById('result-high').textContent   = fmt(data.price_range.high);
  document.getElementById('result-confidence').textContent = data.confidence_note;

  const kwBlock = document.getElementById('keywords-block');
  const kwList  = document.getElementById('keywords-list');
  if (data.keywords_detected && data.keywords_detected.length > 0) {
    kwList.innerHTML = data.keywords_detected
      .map(k => `<span class="kw-badge">${formatKeyword(k)}</span>`)
      .join('');
    kwBlock.classList.remove('hidden');
  }

  const warnBlock = document.getElementById('warnings-block');
  const warnList  = document.getElementById('warnings-list');
  if (data.warnings && data.warnings.length > 0) {
    warnList.innerHTML = data.warnings.map(w => `<li>${w}</li>`).join('');
    warnBlock.classList.remove('hidden');
  }

  resultPanel.classList.remove('hidden');
  resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Helpers ───────────────────────────────────────────────────
function fmt(n) {
  return Number(n).toLocaleString('en-CA', { minimumFractionDigits: 0, maximumFractionDigits: 0 });
}

function formatKeyword(raw) {
  // kw_air_conditioning → Air Conditioning
  return raw.replace(/^kw_/, '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function setLoading(on) {
  submitBtn.disabled = on;
  btnText.classList.toggle('hidden', on);
  btnSpinner.classList.toggle('hidden', !on);
}

function clearResults() {
  resultPanel.classList.add('hidden');
  errorBanner.classList.add('hidden');
  errorBanner.textContent = '';
  document.getElementById('keywords-block').classList.add('hidden');
  document.getElementById('warnings-block').classList.add('hidden');
  document.getElementById('keywords-list').innerHTML = '';
  document.getElementById('warnings-list').innerHTML = '';
}

function showError(msg) {
  errorBanner.textContent = msg;
  errorBanner.classList.remove('hidden');
}

// ── Init ──────────────────────────────────────────────────────
loadNeighbourhoods();
