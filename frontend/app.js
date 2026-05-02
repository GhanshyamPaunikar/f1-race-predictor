const { createApp, ref, computed, watch, nextTick, onMounted } = Vue;

const API = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
  ? 'http://localhost:8001'
  : '';

// Country → emoji flag helper
const COUNTRY_FLAGS = {
  'Australia': '🇦🇺', 'Bahrain': '🇧🇭', 'Saudi Arabia': '🇸🇦', 'Japan': '🇯🇵',
  'China': '🇨🇳', 'USA': '🇺🇸', 'United States': '🇺🇸', 'Italy': '🇮🇹',
  'Monaco': '🇲🇨', 'Canada': '🇨🇦', 'Spain': '🇪🇸', 'Austria': '🇦🇹',
  'UK': '🇬🇧', 'United Kingdom': '🇬🇧', 'Hungary': '🇭🇺', 'Belgium': '🇧🇪',
  'Netherlands': '🇳🇱', 'Singapore': '🇸🇬', 'Azerbaijan': '🇦🇿', 'Mexico': '🇲🇽',
  'Brazil': '🇧🇷', 'UAE': '🇦🇪', 'Abu Dhabi': '🇦🇪', 'Qatar': '🇶🇦',
  'France': '🇫🇷', 'Germany': '🇩🇪', 'Russia': '🇷🇺', 'Turkey': '🇹🇷',
  'Portugal': '🇵🇹', 'Bahrain': '🇧🇭', 'Miami': '🇺🇸', 'Las Vegas': '🇺🇸',
  'Mexico City': '🇲🇽',
};

const FEATURE_LABELS = {
  grid_position:       'Starting Grid Position',
  team_pts_pct:        'Team Points Share (%)',
  driver_season_pts:   'Driver Season Points',
  recent5_avg:         'Recent Form (last 5)',
  recent3_avg:         'Recent Form (last 3)',
  consistency:         'Finishing Consistency',
  circuit_avg:         'Circuit History Avg',
  circuit_exp:         'Circuit Experience',
  season_progress:     'Season Progress',
  is_front_row:        'Front Row Qualifier',
  is_top5_grid:        'Top 5 Grid Position',
  grid_vs_form:        'Grid vs. Expected Form',
};

createApp({
  setup() {
    // ── State ──────────────────────────────────────────────
    const currentYear = ref(2025);
    const activeTab   = ref('predictions');
    const schedule    = ref([]);
    const selectedRound = ref(null);
    const prediction    = ref(null);
    const driverStandings     = ref([]);
    const constructorStandings = ref([]);
    const modelMetrics  = ref(null);
    const trainingState = ref({ state: 'idle', progress: 0, message: '' });

    const loadingPrediction = ref(false);
    const loadingStandings  = ref(false);

    let statusInterval = null;
    let featureChart   = null;

    // ── Computed ───────────────────────────────────────────
    const tabs = [
      { id: 'predictions', label: 'PREDICTIONS' },
      { id: 'standings',   label: 'STANDINGS'   },
      { id: 'model',       label: 'MODEL STATS'  },
    ];

    const isTraining = computed(() =>
      ['fetching', 'training'].includes(trainingState.value.state)
    );
    const trainingProgress = computed(() => trainingState.value.progress ?? 0);
    const trainingMessage  = computed(() => trainingState.value.message ?? '');

    const modelStatusClass = computed(() => ({
      ready:    trainingState.value.state === 'done',
      training: isTraining.value,
      error:    trainingState.value.state === 'error',
    }));
    const modelStatusLabel = computed(() => {
      switch (trainingState.value.state) {
        case 'done':     return 'MODEL READY';
        case 'fetching': return 'FETCHING DATA';
        case 'training': return 'TRAINING';
        case 'error':    return 'ERROR';
        default:         return 'LOADING';
      }
    });

    const sortedFeatures = computed(() => {
      const fi = modelMetrics.value?.feature_importance;
      if (!fi) return {};
      const entries = Object.entries(fi).sort((a, b) => b[1] - a[1]);
      return Object.fromEntries(entries);
    });

    const modelInfo = computed(() => [
      { label: 'Architecture',      value: 'XGBoost + HistGBR + GBR Ensemble' },
      { label: 'Ensemble Weights',  value: '45% XGB · 35% HistGBR · 20% GBR' },
      { label: 'Training Years',    value: '2018 – 2025' },
      { label: 'Features',          value: '12 engineered features' },
      { label: 'Validation',        value: '5-Fold Cross-Validation' },
      { label: 'Simulation',        value: '1000× Monte Carlo win probabilities' },
      { label: 'Hyperparameters',   value: '400 trees, depth 5, lr 0.04' },
      { label: 'Rolling Windows',   value: 'Last 3 & last 5 races (form)' },
    ]);

    // ── Methods ────────────────────────────────────────────
    async function apiFetch(path) {
      const res = await fetch(API + path);
      if (!res.ok) throw new Error(`API error ${res.status} for ${path}`);
      return res.json();
    }

    async function pollStatus() {
      try {
        const data = await apiFetch('/api/status');
        trainingState.value = data.training ?? { state: 'idle', progress: 0, message: '' };
        if (data.metrics) modelMetrics.value = data.metrics;
        if (data.training?.state === 'done' && statusInterval) {
          clearInterval(statusInterval);
          statusInterval = null;
        }
      } catch (e) { /* silent */ }
    }

    async function loadSchedule() {
      try {
        schedule.value = await apiFetch(`/api/schedule/${currentYear.value}`);
        // Auto-select first upcoming race, or last race
        const upcoming = schedule.value.find(r => r.is_upcoming);
        const target = upcoming ?? schedule.value[schedule.value.length - 1];
        if (target) selectRace(target.round);
      } catch (e) { console.error('Schedule error', e); }
    }

    async function selectRace(round) {
      selectedRound.value = round;
      loadingPrediction.value = true;
      prediction.value = null;
      try {
        prediction.value = await apiFetch(`/api/predict/${currentYear.value}/${round}`);
        await nextTick();
        renderFeatureChart();
      } catch (e) {
        console.error('Prediction error', e);
      } finally {
        loadingPrediction.value = false;
      }
    }

    async function loadStandings() {
      loadingStandings.value = true;
      try {
        [driverStandings.value, constructorStandings.value] = await Promise.all([
          apiFetch(`/api/standings/drivers/${currentYear.value}`),
          apiFetch(`/api/standings/constructors/${currentYear.value}`),
        ]);
      } catch (e) { console.error('Standings error', e); }
      finally { loadingStandings.value = false; }
    }

    async function retrain() {
      await fetch(API + '/api/train', { method: 'POST' });
      trainingState.value = { state: 'fetching', progress: 5, message: 'Starting…' };
      startStatusPoll();
    }

    function startStatusPoll() {
      if (statusInterval) clearInterval(statusInterval);
      statusInterval = setInterval(pollStatus, 2000);
    }

    function changeYear(delta) {
      const next = currentYear.value + delta;
      if (next >= 2018 && next <= 2026) {
        currentYear.value = next;
      }
    }

    // ── Formatting helpers ─────────────────────────────────
    function formatDate(d) {
      if (!d) return '—';
      const dt = new Date(d + 'T00:00:00');
      return dt.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' });
    }

    function countryFlag(country) {
      return COUNTRY_FLAGS[country] ?? '🏁';
    }

    function driverInitials(name) {
      if (!name) return '??';
      const parts = name.trim().split(' ');
      if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
      return (parts[0][0] + parts[parts.length - 1].slice(0, 2)).toUpperCase();
    }

    function shortName(name) {
      if (!name) return '';
      const parts = name.split(' ');
      if (parts.length < 2) return name;
      return parts[parts.length - 1].toUpperCase();
    }

    function pct(v) {
      return (Math.round((v ?? 0) * 1000) / 10).toFixed(1);
    }

    function posClass(pos) {
      if (pos === 1) return 'p1';
      if (pos === 2) return 'p2';
      if (pos === 3) return 'p3';
      if (pos <= 10) return 'points';
      return '';
    }

    function confColor(c) {
      if (c >= 0.75) return '#00c950';
      if (c >= 0.5)  return '#f90';
      return '#e8002d';
    }

    function featureLabel(key) {
      return FEATURE_LABELS[key] ?? key;
    }

    function isCorrect(p) {
      const actual = prediction.value?.actual_results?.[p.driver_id];
      return actual && actual === p.predicted_position;
    }

    function isWrong(p) {
      const actual = prediction.value?.actual_results?.[p.driver_id];
      return actual && Math.abs(actual - p.predicted_position) > 4;
    }

    // ── Chart ──────────────────────────────────────────────
    function renderFeatureChart() {
      const fi = prediction.value?.feature_importance;
      if (!fi) return;
      const canvas = document.getElementById('featureChart');
      if (!canvas) return;

      if (featureChart) featureChart.destroy();

      const sorted = Object.entries(fi).sort((a, b) => b[1] - a[1]);
      const labels = sorted.map(([k]) => FEATURE_LABELS[k] ?? k);
      const values = sorted.map(([, v]) => +(v * 100).toFixed(2));

      featureChart = new Chart(canvas, {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            data: values,
            backgroundColor: values.map((_, i) =>
              i === 0 ? '#e8002d' : i < 3 ? '#e8002d88' : '#e8002d44'
            ),
            borderRadius: 4,
          }],
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: {
              ticks: { color: '#666', font: { size: 9 } },
              grid:  { color: '#1f1f1f' },
            },
            y: {
              ticks: { color: '#aaa', font: { size: 10 } },
              grid:  { display: false },
            },
          },
        },
      });
    }

    // ── Watchers ───────────────────────────────────────────
    watch(currentYear, () => {
      schedule.value = [];
      prediction.value = null;
      driverStandings.value = [];
      constructorStandings.value = [];
      loadSchedule();
    });

    watch(activeTab, (tab) => {
      if (tab === 'standings' && !driverStandings.value.length) loadStandings();
      if (tab === 'model') pollStatus();
    });

    // ── Lifecycle ──────────────────────────────────────────
    onMounted(async () => {
      await pollStatus();
      startStatusPoll();
      await loadSchedule();
    });

    return {
      // state
      currentYear, activeTab, schedule, selectedRound,
      prediction, driverStandings, constructorStandings,
      modelMetrics, trainingState, loadingPrediction, loadingStandings,
      // computed
      tabs, isTraining, trainingProgress, trainingMessage,
      modelStatusClass, modelStatusLabel, sortedFeatures, modelInfo,
      // methods
      selectRace, changeYear, retrain,
      // formatters
      formatDate, countryFlag, driverInitials, shortName,
      pct, posClass, confColor, featureLabel, isCorrect, isWrong,
    };
  },
}).mount('#app');
