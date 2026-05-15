const { createApp, ref, computed, watch, nextTick, onMounted } = Vue;

const API = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
  ? 'http://localhost:8001'
  : '';

const COUNTRY_FLAGS = {
  'Australia': '🇦🇺', 'Bahrain': '🇧🇭', 'Saudi Arabia': '🇸🇦', 'Japan': '🇯🇵',
  'China': '🇨🇳', 'USA': '🇺🇸', 'United States': '🇺🇸', 'Italy': '🇮🇹',
  'Monaco': '🇲🇨', 'Canada': '🇨🇦', 'Spain': '🇪🇸', 'Austria': '🇦🇹',
  'UK': '🇬🇧', 'United Kingdom': '🇬🇧', 'Hungary': '🇭🇺', 'Belgium': '🇧🇪',
  'Netherlands': '🇳🇱', 'Singapore': '🇸🇬', 'Azerbaijan': '🇦🇿', 'Mexico': '🇲🇽',
  'Brazil': '🇧🇷', 'UAE': '🇦🇪', 'Abu Dhabi': '🇦🇪', 'Qatar': '🇶🇦',
  'France': '🇫🇷', 'Germany': '🇩🇪', 'Russia': '🇷🇺', 'Turkey': '🇹🇷',
  'Portugal': '🇵🇹', 'Miami': '🇺🇸', 'Las Vegas': '🇺🇸', 'Mexico City': '🇲🇽',
};

// Friendly labels for every feature the backend may surface. Keep in sync with
// the FEATURE_LABELS dict in backend/predictor.py — that one is authoritative
// and is also returned by /api/model/accuracy.
const FEATURE_LABELS = {
  grid_position:        'Starting Grid Position',
  inv_grid:             'Inverse Grid (1/pos)',
  exp_grid:             'Grid Decay (exp)',
  is_front_row:         'Front Row Qualifier',
  team_pts_pct:         'Team Points Share (%)',
  team_form_last3:      'Team Form (last 3 races)',
  driver_season_pts:    'Driver Season Points',
  recent5_avg:          'Recent Form (last 5)',
  recent3_avg:          'Recent Form (last 3)',
  consistency:          'Finishing Consistency',
  circuit_avg:          'Circuit History Avg',
  circuit_exp:          'Circuit Experience',
  season_progress:      'Season Progress',
  grid_vs_form:         'Grid vs. Expected Form',
  quali_gap_to_pole:    'Quali Gap to Pole (s)',
  air_temp_c:           'Air Temperature (°C)',
  precip_mm:            'Precipitation (mm)',
  fp_best_lap_norm:     'Practice Best Lap Gap (s)',
  fp_longrun_pace_norm: 'Practice Long-Run Pace Gap (s)',
  fp_teammate_quali_gap:'Quali Gap vs Teammate (s)',
  fp_session_laps:      'Practice Laps Completed',
  fp_consistency:       'Practice Lap-Time Consistency',
};

createApp({
  setup() {
    // ── State ─────────────────────────────────────────────────
    const currentYear   = ref(2025);
    const activeTab     = ref('predictions');
    const schedule      = ref([]);
    const selectedRound = ref(null);
    const prediction    = ref(null);
    const driverStandings      = ref([]);
    const constructorStandings = ref([]);
    const modelMetrics  = ref(null);
    const trainingState = ref({ state: 'idle', progress: 0, message: '' });
    const historyData   = ref(null);
    const explainData   = ref(null);
    const compareData   = ref(null);
    const compareSelection = ref([]);
    const theme         = ref(localStorage.getItem('f1-theme') || 'dark');

    const loadingPrediction = ref(false);
    const loadingStandings  = ref(false);
    const loadingHistory    = ref(false);
    const loadingCompare    = ref(false);

    let statusInterval = null;
    let featureChart   = null;
    let historyChart   = null;

    // ── Tabs / status ─────────────────────────────────────────
    const tabs = [
      { id: 'predictions', label: 'PREDICTIONS' },
      { id: 'compare',     label: 'COMPARE'     },
      { id: 'history',     label: 'TRACK RECORD'},
      { id: 'standings',   label: 'STANDINGS'   },
      { id: 'model',       label: 'MODEL STATS' },
    ];
    const isTraining = computed(() =>
      ['fetching', 'training'].includes(trainingState.value.state));
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
      // Drop features below 1% — their bars are unreadable and clutter the panel
      const entries = Object.entries(fi)
        .filter(([, v]) => v >= 0.01)
        .sort((a, b) => b[1] - a[1]);
      return Object.fromEntries(entries);
    });

    // The first race in the schedule that hasn't happened yet
    const nextUpcomingRound = computed(() => {
      const next = schedule.value.find(r => r.is_upcoming);
      return next ? next.round : null;
    });
    const modelInfo = computed(() => [
      { label: 'Architecture',      value: 'XGBoost + HistGBR + GBR Ensemble' },
      { label: 'Quantile Heads',    value: 'P10 / P90 (calibrated uncertainty)' },
      { label: 'DNF Head',          value: 'GradientBoosting classifier' },
      { label: 'Validation',        value: 'GroupKFold by season (no leakage)' },
      { label: 'Features',          value: '14 (incl. weather)' },
      { label: 'Simulation',        value: '2000× Monte Carlo with DNF sampling' },
      { label: 'Training Years',    value: '2018 – 2025' },
      { label: 'Hyperparameters',   value: '500 trees, depth 5, lr 0.035' },
    ]);

    // ── HTTP ──────────────────────────────────────────────────
    async function apiFetch(path) {
      const res = await fetch(API + path);
      if (!res.ok) throw new Error(`API ${res.status} for ${path}`);
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
        const upcoming = schedule.value.find(r => r.is_upcoming);
        const target = upcoming ?? schedule.value[schedule.value.length - 1];
        if (target) selectRace(target.round);
      } catch (e) { console.error('Schedule error', e); }
    }

    async function selectRace(round) {
      selectedRound.value = round;
      loadingPrediction.value = true;
      prediction.value = null;
      explainData.value = null;
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

    async function loadHistory() {
      loadingHistory.value = true;
      historyData.value = null;
      try {
        historyData.value = await apiFetch(`/api/predictions/history/${currentYear.value}`);
      } catch (e) { console.error('History error', e); }
      finally {
        loadingHistory.value = false;
        // Canvas only exists after loadingHistory flips false and Vue re-renders
        await nextTick();
        renderHistoryChart();
      }
    }

    async function loadCompare() {
      if (compareSelection.value.length < 2) return;
      if (selectedRound.value == null) return;
      loadingCompare.value = true;
      try {
        const ids = compareSelection.value.join(',');
        compareData.value = await apiFetch(
          `/api/compare/${currentYear.value}/${selectedRound.value}?drivers=${ids}`
        );
      } catch (e) { console.error('Compare error', e); }
      finally { loadingCompare.value = false; }
    }

    function toggleCompare(driverId) {
      const i = compareSelection.value.indexOf(driverId);
      if (i >= 0) compareSelection.value.splice(i, 1);
      else if (compareSelection.value.length < 4) compareSelection.value.push(driverId);
    }

    async function explainDriver(driverId) {
      try {
        explainData.value = { loading: true, driver_id: driverId };
        const d = await apiFetch(
          `/api/explain/${currentYear.value}/${selectedRound.value}/${driverId}`
        );
        const p = prediction.value?.predictions.find(p => p.driver_id === driverId);
        explainData.value = { ...d, name: p?.name, team: p?.team, team_color: p?.team_color,
          predicted_position: p?.predicted_position };
      } catch (e) {
        explainData.value = { error: e.message };
      }
    }
    function closeExplain() { explainData.value = null; }

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
      if (next >= 2018 && next <= 2026) currentYear.value = next;
    }

    function toggleTheme() {
      theme.value = theme.value === 'dark' ? 'light' : 'dark';
      localStorage.setItem('f1-theme', theme.value);
      document.documentElement.setAttribute('data-theme', theme.value);
    }

    // ── Formatting ────────────────────────────────────────────
    function formatDate(d) {
      if (!d) return '—';
      const dt = new Date(d + 'T00:00:00');
      return dt.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' });
    }
    function countryFlag(c) { return COUNTRY_FLAGS[c] ?? '🏁'; }
    // Prefer official F1 three-letter code (VER, NOR, RUS) when available,
    // fall back to a heuristic from the driver's name.
    function driverInitials(nameOrDriver) {
      if (nameOrDriver && typeof nameOrDriver === 'object') {
        if (nameOrDriver.code) return nameOrDriver.code.toUpperCase();
        nameOrDriver = nameOrDriver.name;
      }
      if (!nameOrDriver) return '??';
      const parts = nameOrDriver.trim().split(' ');
      if (parts.length === 1) return parts[0].slice(0, 3).toUpperCase();
      return (parts[0][0] + parts[parts.length - 1].slice(0, 2)).toUpperCase();
    }
    function shortName(name) {
      if (!name) return '';
      const parts = name.split(' ');
      if (parts.length < 2) return name;
      return parts[parts.length - 1].toUpperCase();
    }
    function pct(v) { return (Math.round((v ?? 0) * 1000) / 10).toFixed(1); }
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
    function dnfColor(d) {
      if (d >= 0.35) return '#e8002d';
      if (d >= 0.2)  return '#f90';
      return '#666';
    }
    function featureLabel(k) {
      return modelMetrics.value?.feature_labels?.[k]
        ?? FEATURE_LABELS[k]
        ?? k;
    }
    function isCorrect(p) {
      const a = prediction.value?.actual_results?.[p.driver_id];
      return a && a === p.predicted_position;
    }
    function isWrong(p) {
      const a = prediction.value?.actual_results?.[p.driver_id];
      return a && Math.abs(a - p.predicted_position) > 4;
    }
    function inCompare(id) { return compareSelection.value.includes(id); }

    // ── Charts ────────────────────────────────────────────────
    function chartColors() {
      const dark = theme.value === 'dark';
      return {
        grid:  dark ? '#1f1f1f' : '#e5e5e5',
        tick:  dark ? '#aaa'    : '#555',
        muted: dark ? '#666'    : '#999',
        red:   '#e8002d',
        green: '#00c950',
      };
    }

    function renderFeatureChart() {
      const fi = prediction.value?.feature_importance;
      if (!fi) return;
      const canvas = document.getElementById('featureChart');
      if (!canvas) return;
      if (featureChart) featureChart.destroy();
      const sorted = Object.entries(fi).sort((a, b) => b[1] - a[1]);
      const labels = sorted.map(([k]) => FEATURE_LABELS[k] ?? k);
      const values = sorted.map(([, v]) => +(v * 100).toFixed(2));
      const c = chartColors();
      featureChart = new Chart(canvas, {
        type: 'bar',
        data: { labels, datasets: [{
          data: values,
          backgroundColor: values.map((_, i) => i === 0 ? c.red : i < 3 ? c.red + '88' : c.red + '44'),
          borderRadius: 4,
        }]},
        options: {
          indexAxis: 'y',
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { ticks: { color: c.muted, font: { size: 9 } }, grid: { color: c.grid } },
            y: { ticks: { color: c.tick, font: { size: 10 } }, grid: { display: false } },
          },
        },
      });
    }

    function renderHistoryChart() {
      const canvas = document.getElementById('historyChart');
      if (!canvas || !historyData.value) return;
      if (historyChart) historyChart.destroy();
      const races = historyData.value.races;
      const c = chartColors();
      historyChart = new Chart(canvas, {
        type: 'bar',
        data: {
          labels: races.map(r => `R${r.round} ${countryFlag(r.country)}`),
          datasets: [
            {
              label: 'Podium drivers correctly predicted (out of 3)',
              data: races.map(r => r.podium_hits),
              backgroundColor: c.red + 'cc', borderRadius: 3,
            },
            {
              label: 'Race winner predicted correctly (1 = yes, 0 = no)',
              data: races.map(r => r.winner_hit ? 1 : 0),
              backgroundColor: c.green + 'cc', borderRadius: 3,
            },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: {
            legend: { labels: { color: c.tick, font: { size: 11 }, boxWidth: 14 } },
            tooltip: {
              callbacks: {
                title: (items) => {
                  const r = races[items[0].dataIndex];
                  return `Round ${r.round}: ${r.name}`;
                },
                label: (item) => {
                  const r = races[item.dataIndex];
                  if (item.datasetIndex === 0) {
                    return `Podium: ${r.podium_hits}/3 drivers correct`;
                  }
                  return r.winner_hit
                    ? `Winner: correct (${r.predicted_winner})`
                    : `Winner: missed (predicted ${r.predicted_winner ?? '—'}, actual ${r.actual_winner ?? '—'})`;
                },
              },
            },
          },
          scales: {
            x: { ticks: { color: c.muted, font: { size: 9 } }, grid: { color: c.grid } },
            y: {
              beginAtZero: true, max: 3,
              ticks: { color: c.muted, stepSize: 1 },
              grid: { color: c.grid },
              title: { display: true, text: 'Correct predictions', color: c.muted, font: { size: 11 } },
            },
          },
        },
      });
    }

    // ── Watchers ──────────────────────────────────────────────
    watch(currentYear, () => {
      schedule.value = [];
      prediction.value = null;
      driverStandings.value = [];
      constructorStandings.value = [];
      historyData.value = null;
      compareData.value = null;
      compareSelection.value = [];
      loadSchedule();
    });

    watch(activeTab, (tab) => {
      if (tab === 'standings' && !driverStandings.value.length) loadStandings();
      if (tab === 'model')   pollStatus();
      if (tab === 'history' && !historyData.value) loadHistory();
    });

    watch(theme, () => {
      document.documentElement.setAttribute('data-theme', theme.value);
      nextTick(() => { renderFeatureChart(); renderHistoryChart(); });
    });

    // ── Lifecycle ─────────────────────────────────────────────
    onMounted(async () => {
      document.documentElement.setAttribute('data-theme', theme.value);
      await pollStatus();
      startStatusPoll();
      await loadSchedule();
    });

    return {
      // state
      currentYear, activeTab, schedule, selectedRound,
      prediction, driverStandings, constructorStandings,
      modelMetrics, trainingState, loadingPrediction, loadingStandings,
      historyData, loadingHistory, compareData, compareSelection, loadingCompare,
      explainData, theme,
      // computed
      tabs, isTraining, trainingProgress, trainingMessage,
      modelStatusClass, modelStatusLabel, sortedFeatures, modelInfo, nextUpcomingRound,
      // methods
      selectRace, changeYear, retrain, loadCompare, toggleCompare,
      explainDriver, closeExplain, toggleTheme, inCompare,
      // formatters
      formatDate, countryFlag, driverInitials, shortName,
      pct, posClass, confColor, dnfColor, featureLabel, isCorrect, isWrong,
    };
  },
}).mount('#app');
