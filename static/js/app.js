/* =====================================================================
   PetIA Shop · Lógica del dashboard  (Integrante 1)
   - Navegación entre vistas (los paneles existen una sola vez y se mueven)
   - Selector de mascota con imagen vinculada a la raza
   - Formulario de 5 rasgos, alta de registros y validación
   - Carga de CSV con progreso e informe de validación
   - Tabla paginada, gráficas del dataset y diccionario de datos
   - Renderizado de resultados de entrenamiento / clasificación / K-means /
     Apriori según docs/CONTRATO_API.md (los implementan Integrantes 2 y 3)
   ===================================================================== */
(() => {
  "use strict";

  // ------------------------------------------------------------ utilidades
  const $ = (s, c = document) => c.querySelector(s);
  const $$ = (s, c = document) => [...c.querySelectorAll(s)];
  const fmt = (n, d = 0) => Number(n).toLocaleString("es-MX", { minimumFractionDigits: d, maximumFractionDigits: d });
  const f3 = (n) => (n == null || isNaN(n) ? "—" : Number(n).toFixed(3));
  const pct = (n) => (Number(n) * 100).toFixed(1) + " %";
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const COLOR_TIPO = { Perro: "#2563eb", Gato: "#8b5cf6", Ave: "#10b981", Tortuga: "#f5b50b" };
  const EMOJI_TIPO = { Perro: "🐕", Gato: "🐈", Ave: "🦜", Tortuga: "🐢" };
  const COLOR_GRUPO = ["#2563eb", "#8b5cf6", "#10b981", "#f5b50b", "#ef4444", "#06b6d4", "#ec4899", "#64748b"];
  const NOMBRE_RASGO = { peso_kg: "Peso (kg)", altura_cm: "Altura (cm)", edad_anios: "Edad (años)", nivel_actividad: "Nivel de actividad", tipo_pelaje: "Tipo de pelaje", tipo: "Tipo", raza: "Raza" };

  function icono(nombre) {
    const p = (window.ICONOS || {})[nombre];
    return p ? `<svg class="i" viewBox="0 0 24 24" aria-hidden="true">${p}</svg>` : "";
  }
  function hidratarIconos(raiz = document) {
    $$("i[data-i]", raiz).forEach((el) => {
      const tmp = document.createElement("span");
      tmp.innerHTML = icono(el.dataset.i);
      const svg = tmp.firstElementChild;
      if (!svg) return;
      if (el.id) svg.id = el.id;
      el.className && svg.classList.add(...el.className.split(" ").filter(Boolean));
      el.replaceWith(svg);
    });
  }

  async function api(url, opciones = {}) {
    const op = { headers: {}, ...opciones };
    if (op.json !== undefined) {
      op.method = op.method || "POST";
      op.headers["Content-Type"] = "application/json";
      op.body = JSON.stringify(op.json);
      delete op.json;
    }
    try {
      const r = await fetch(url, op);
      let datos = null;
      try { datos = await r.json(); } catch { /* sin cuerpo JSON */ }
      return { ok: r.ok, status: r.status, datos };
    } catch (e) {
      return { ok: false, status: 0, datos: { mensaje: "No hay conexión con el servidor Flask." } };
    }
  }

  function toast(titulo, texto = "", tipo = "info", ms = 4200) {
    const ico = { ok: "circle-check", error: "circle-x", aviso: "triangle-alert", info: "info" }[tipo];
    const t = document.createElement("div");
    t.className = `toast ${tipo}`;
    t.innerHTML = `${icono(ico)}<div><b>${esc(titulo)}</b>${esc(texto)}</div>`;
    $("#toasts").appendChild(t);
    setTimeout(() => { t.classList.add("sale"); setTimeout(() => t.remove(), 300); }, ms);
  }

  function cargando(botones, si) {
    botones.forEach((b) => { b.classList.toggle("cargando", si); b.disabled = si; });
  }

  // ------------------------------------------------------------ estado
  const S = {
    esquema: null,
    resumen: null,
    tabla: { pagina: 1, por: 15, tipo: "", raza: "", buscar: "", orden: "id", desc: true, resaltar: null },
    graficas: {},
    imagenes: {},
    ultimaPrediccion: null,
    matrices: null,
    nivelMatriz: "tipo",
    resultadosEntrenamiento: null,
  };

  // =====================================================================
  //  NAVEGACIÓN
  // =====================================================================
  const VISTAS = {
    inicio: ["", ""],
    dataset: ["Administración del dataset", "Carga CSV, validación de campos, alta de mascotas, registros y diccionario de datos."],
    entrenamiento: ["Entrenamiento", "División 75/15/10 sin fugas y comparación de GaussianNB, Árbol de Decisión y Random Forest."],
    clasificacion: ["Clasificación", "Ingresa los cinco rasgos: el modelo predice primero el tipo y después la raza, con sus probabilidades."],
    agrupacion: ["Agrupación (K-means)", "Perfiles semejantes descubiertos sin usar las etiquetas como entrada."],
    asociacion: ["Asociación (Apriori)", "Reglas entre características y categorías con soporte, confianza y lift."],
    resultados: ["Resultados", "Métricas finales en el conjunto de prueba, matriz de confusión y comparación de modelos."],
    acerca: ["Acerca de", "Información del proyecto y del equipo."],
  };

  function irA(vista) {
    if (!VISTAS[vista]) vista = "inicio";
    const sec = $(`.vista[data-vista="${vista}"]`);
    $$(".vista").forEach((v) => v.classList.toggle("activa", v === sec));
    $$("[data-slot]", sec).forEach((slot) => {
      const card = document.querySelector(`[data-card="${slot.dataset.slot}"]`);
      if (card && card.parentElement !== slot) slot.appendChild(card);
    });
    $$(".nav a").forEach((a) => a.classList.toggle("activo", a.dataset.vista === vista));
    const [titulo, desc] = VISTAS[vista];
    $("#vista-cabecera").classList.toggle("visible", !!titulo);
    $("#vista-titulo").textContent = titulo;
    $("#vista-desc").textContent = desc;
    document.body.classList.remove("menu-abierto");
    // las gráficas se reacomodan al cambiar de contenedor
    requestAnimationFrame(() => Object.values(S.graficas).forEach((g) => g && g.resize()));
    // Repintar SIEMPRE los resultados del objetivo seleccionado al cambiar de vista
    if (S.resultadosEntrenamiento) {
      pintarResultadosNivel();
    }
    if (vista === "dataset") cargarTabla();
  }

  function iniciarNavegacion() {
    window.addEventListener("hashchange", () => irA(location.hash.slice(1)));
    $("#btn-menu").addEventListener("click", () => document.body.classList.toggle("menu-abierto"));
    $("#velo").addEventListener("click", () => document.body.classList.remove("menu-abierto"));
    irA(location.hash.slice(1) || "inicio");
  }

  // =====================================================================
  //  IMÁGENES VINCULADAS A LA RAZA
  //  1) static/img/razas/<slug>.jpg  2) Wikipedia (en línea)  3) ilustración
  // =====================================================================
  function ilustracion(raza) {
    const tipo = S.esquema?.razas[raza]?.tipo || "Perro";
    const c = COLOR_TIPO[tipo];
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 410">
      <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="${c}" stop-opacity=".22"/><stop offset="1" stop-color="${c}" stop-opacity=".55"/></linearGradient></defs>
      <rect width="400" height="410" fill="url(#g)"/><circle cx="200" cy="180" r="120" fill="#fff" fill-opacity=".55"/>
      <text x="200" y="228" font-size="140" text-anchor="middle">${EMOJI_TIPO[tipo]}</text>
      <text x="200" y="350" font-family="Segoe UI,Arial" font-weight="700" font-size="26" fill="#0e2a57" text-anchor="middle">${esc(raza)}</text></svg>`;
    return "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
  }

  async function urlImagen(raza) {
    if (S.imagenes[raza]) return S.imagenes[raza];
    const local = S.esquema?.imagenes_locales?.[raza];
    if (local) return (S.imagenes[raza] = local);
    let guardada = null;
    try { guardada = localStorage.getItem("img:" + raza); } catch { /* almacenamiento no disponible */ }
    if (guardada) return (S.imagenes[raza] = guardada);
    const wiki = S.esquema?.razas[raza]?.wiki;
    if (wiki && navigator.onLine !== false) {
      try {
        const ctrl = new AbortController();
        const t = setTimeout(() => ctrl.abort(), 5000);
        const r = await fetch(`https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(wiki)}`, { signal: ctrl.signal });
        clearTimeout(t);
        if (r.ok) {
          const j = await r.json();
          let src = j.thumbnail?.source || j.originalimage?.source;
          if (src) {
            src = src.replace(/\/(\d+)px-/, "/640px-");
            try { localStorage.setItem("img:" + raza, src); } catch { /* sin almacenamiento */ }
            return (S.imagenes[raza] = src);
          }
        }
      } catch { /* sin internet: se usa la ilustración */ }
    }
    return (S.imagenes[raza] = ilustracion(raza));
  }

  async function ponerImagen(img, raza) {
    const url = await urlImagen(raza);
    if (img.dataset.src === url) return;
    img.classList.add("cambiando");
    const pre = new Image();
    pre.onload = () => { img.src = url; img.dataset.src = url; img.classList.remove("cambiando"); };
    pre.onerror = () => { S.imagenes[raza] = ilustracion(raza); img.src = S.imagenes[raza]; img.classList.remove("cambiando"); };
    pre.src = url;
    img.alt = `Fotografía de ${raza}`;
  }

  // =====================================================================
  //  SELECTOR DE MASCOTA Y FORMULARIO DE 5 RASGOS
  // =====================================================================
  const F = {
    form: () => $("#form-mascota"), tipo: () => $("#sel-tipo"), raza: () => $("#sel-raza"),
    peso: () => $("#in-peso"), altura: () => $("#in-altura"), edad: () => $("#in-edad"),
    actividad: () => $("#in-actividad"), pelaje: () => $("#in-pelaje"),
  };

  function llenarSelect(sel, valores, actual) {
    sel.innerHTML = valores.map((v) => `<option value="${esc(v)}">${esc(v)}</option>`).join("");
    if (actual && valores.includes(actual)) sel.value = actual;
  }

  function iniciarSelector() {
    const E = S.esquema;
    llenarSelect(F.tipo(), E.tipos, "Perro");
    llenarSelect(F.actividad(), E.niveles_actividad, "Alta");
    llenarSelect(F.pelaje(), E.tipos_pelaje, "Corto");
    actualizarRazas("Labrador Retriever");

    F.tipo().addEventListener("change", () => { actualizarRazas(); autocompletar(true); });
    F.raza().addEventListener("change", () => { vistaPrevia(); autocompletar(true); });
    $("#btn-autocompletar").addEventListener("click", () => autocompletar(false));
    F.form().addEventListener("submit", (e) => { e.preventDefault(); anadirRegistro(); });
    F.form().addEventListener("input", (e) => limpiarError(e.target.name));
    F.form().addEventListener("change", (e) => limpiarError(e.target.name));
    $("#btn-cargar").addEventListener("click", () => {
      location.hash = "dataset";
      setTimeout(() => { $("#card-carga").scrollIntoView({ behavior: "smooth", block: "center" }); $("#dropzone").classList.add("sobre"); setTimeout(() => $("#dropzone").classList.remove("sobre"), 1200); }, 80);
    });
    autocompletar(true);
  }

  function actualizarRazas(preferida) {
    const tipo = F.tipo().value;
    llenarSelect(F.raza(), S.esquema.razas_por_tipo[tipo], preferida);
    const ico = $("#ico-tipo");
    if (ico) ico.style.color = COLOR_TIPO[tipo];
    vistaPrevia();
  }

  function vistaPrevia() {
    const raza = F.raza().value;
    if (S.ultimaPrediccion) return;          // si ya hay predicción, se respeta su imagen
    $("#pred-chip").textContent = "Vista previa";
    $("#pred-chip").classList.remove("predicha");
    $("#pred-cap-raza").textContent = `${F.tipo().value} · ${raza}`;
    ponerImagen($("#pred-img"), raza);
  }

  async function autocompletar(silencioso) {
    const raza = F.raza().value;
    const r = await api(`/api/dataset/perfil/${encodeURIComponent(raza)}`);
    if (!r.ok || !r.datos?.raza) return;
    const p = r.datos;
    F.peso().value = p.peso_kg; F.altura().value = p.altura_cm; F.edad().value = p.edad_anios;
    F.actividad().value = p.nivel_actividad; F.pelaje().value = p.tipo_pelaje;
    $$(".err").forEach((e) => (e.textContent = ""));
    $$(".invalido").forEach((e) => e.classList.remove("invalido"));
    if (!silencioso) toast("Valores típicos cargados", ` ${raza}: mediana de ${fmt(p.n)} registros.`, "info", 2600);
  }

  function leerFormulario() {
    const fd = new FormData(F.form());
    return Object.fromEntries(fd.entries());
  }

  /** Validación en el navegador (la definitiva se repite en el servidor). */
  function validarCliente(d, conEtiquetas) {
    const E = S.esquema, err = {};
    for (const c of E.numericas) {
      const v = String(d[c] ?? "").trim().replace(",", ".");
      const [lo, hi] = E.rangos[c];
      if (v === "") err[c] = `${NOMBRE_RASGO[c]} es obligatorio.`;
      else if (isNaN(Number(v))) err[c] = `${NOMBRE_RASGO[c]} debe ser un número.`;
      else if (Number(v) < lo || Number(v) > hi) err[c] = `Debe estar entre ${lo} y ${hi}.`;
      else if (c === "edad_anios" && !Number.isInteger(Number(v))) err[c] = "La edad debe ser un número entero.";
    }
    if (!E.niveles_actividad.includes(d.nivel_actividad)) err.nivel_actividad = "Selecciona un nivel de actividad.";
    if (!E.tipos_pelaje.includes(d.tipo_pelaje)) err.tipo_pelaje = "Selecciona un tipo de pelaje.";
    if (conEtiquetas) {
      if (!E.tipos.includes(d.tipo)) err.tipo = "Selecciona el tipo.";
      if (!E.razas[d.raza]) err.raza = "Selecciona la raza.";
      else if (E.razas[d.raza].tipo !== d.tipo) err.raza = `${d.raza} no es una raza de ${d.tipo}.`;
      if (!err.tipo && !err.tipo_pelaje && !E.pelaje_por_tipo[d.tipo].includes(d.tipo_pelaje))
        err.tipo_pelaje = `Un(a) ${d.tipo} no puede tener pelaje "${d.tipo_pelaje}". Opciones: ${E.pelaje_por_tipo[d.tipo].join(", ")}.`;
    }
    return err;
  }

  function mostrarErrores(err) {
    $$(".err").forEach((e) => (e.textContent = ""));
    $$(".invalido").forEach((e) => e.classList.remove("invalido"));
    Object.entries(err).forEach(([c, m]) => {
      const e = $(`.err[data-err="${c}"]`);
      if (e) e.textContent = m;
      const campo = F.form().elements[c];
      if (campo) campo.classList.add("invalido");
    });
    const primero = Object.keys(err)[0];
    if (primero && F.form().elements[primero]) F.form().elements[primero].focus();
  }
  function limpiarError(nombre) {
    if (!nombre) return;
    const e = $(`.err[data-err="${nombre}"]`);
    if (e) e.textContent = "";
    F.form().elements[nombre]?.classList.remove("invalido");
  }

  async function anadirRegistro() {
    const d = leerFormulario();
    const err = validarCliente(d, true);
    if (Object.keys(err).length) { mostrarErrores(err); toast("Revisa el formulario", " Hay campos con errores.", "error"); return; }
    const btn = $("#btn-anadir");
    cargando([btn], true);
    const r = await api("/api/dataset/registro", { json: d });
    cargando([btn], false);
    if (!r.ok) {
      if (r.datos?.errores) mostrarErrores(r.datos.errores);
      toast("No se añadió el registro", " " + (r.datos?.mensaje || "El servidor rechazó los datos."), "error");
      return;
    }
    toast("¡Mascota añadida!", " " + r.datos.mensaje, "ok");
    Object.assign(S.tabla, { pagina: 1, orden: "id", desc: true, tipo: "", raza: "", buscar: "", resaltar: r.datos.registro.id });
    sincronizarFiltros();
    await refrescarResumen(true);
    cargarTabla();
  }

  // =====================================================================
  //  RESUMEN / KPIs / GRÁFICAS DEL DATASET
  // =====================================================================
  function animarNumero(el, hasta) {
    const desde = Number(el.dataset.valor || 0);
    el.dataset.valor = hasta;
    if (desde === hasta) { el.textContent = fmt(hasta); return; }
    const t0 = performance.now(), dur = desde ? 600 : 1100;
    const paso = (t) => {
      const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      el.textContent = fmt(Math.round(desde + (hasta - desde) * e));
      if (k < 1) requestAnimationFrame(paso);
    };
    requestAnimationFrame(paso);
    if (desde) { el.classList.remove("pulso"); void el.offsetWidth; el.classList.add("pulso"); }
  }

  async function refrescarResumen() {
    const r = await api("/api/dataset/resumen");
    if (!r.ok) { toast("Sin conexión", " No se pudo leer el dataset.", "error"); return; }
    const R = (S.resumen = r.datos);
    animarNumero($("#kpi-total"), R.total);
    $("#kpi-tipos").textContent = R.n_tipos;
    $("#kpi-tipos-txt").textContent = Object.keys(R.por_tipo).filter((t) => R.por_tipo[t] > 0).join(" · ");
    $("#kpi-razas").textContent = R.n_razas;
    $("#kpi-pred").textContent = R.n_predictoras;
    $("#sb-total").textContent = fmt(R.total) + " registros";
    $("#sb-punto").classList.toggle("on", R.total > 0);
    $("#lbl-total-csv").textContent = fmt(R.total);
    // División 75/15/10 calculada sobre el total actual
    const tr = Math.round(R.total * 0.75), va = Math.round(R.total * 0.15);
    if (!S.division) {
      $("#n-train").textContent = fmt(tr); $("#n-val").textContent = fmt(va); $("#n-test").textContent = fmt(R.total - tr - va);
    }
    pintarEstadoModelo(R.modelo);
    pintarGraficasDataset();
  }

  function pintarEstadoModelo(m) {
    const pill = $("#kpi-estado"), ico = $("#kpi-estado-ico");
    let txt = "Sin entrenar", cls = "gris", ic = "circle-alert";
    if (m?.entrenado && m.desactualizado) { txt = "Reentrenar ↻"; cls = "ambar"; ic = "triangle-alert"; }
    else if (m?.entrenado) { txt = "Entrenado ✓"; cls = "verde"; ic = "circle-check"; }
    pill.className = "pill " + cls; pill.textContent = txt;
    ico.className = "kpi-ico " + cls; ico.innerHTML = icono(ic);
    pill.title = m?.entrenado ? `Mejor modelo: ${m.mejor || "—"} · ${m.entrenado_en || ""}` : "Presiona Reentrenar modelos";
  }

  function opcionesBase() {
    return {
      responsive: true, maintainAspectRatio: false, animation: { duration: 700 },
      plugins: { legend: { display: false }, tooltip: { backgroundColor: "#0e2a57", padding: 10, cornerRadius: 8, titleFont: { weight: "700" } } },
      scales: {},
    };
  }
  Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
  Chart.defaults.color = "#4a5b78";

  function pintarGraficasDataset() {
    const R = S.resumen, E = S.esquema;
    // Dona por tipo
    const tipos = E.tipos, vals = tipos.map((t) => R.por_tipo[t] || 0);
    if (!S.graficas.tipos) {
      S.graficas.tipos = new Chart($("#graf-tipos"), {
        type: "doughnut",
        data: { labels: tipos, datasets: [{ data: vals, backgroundColor: tipos.map((t) => COLOR_TIPO[t]), borderColor: "#fff", borderWidth: 3, hoverOffset: 8 }] },
        options: { ...opcionesBase(), cutout: "62%",
          plugins: { legend: { display: true, position: "bottom", labels: { usePointStyle: true, boxWidth: 8, padding: 12 } },
            tooltip: { callbacks: { label: (c) => ` ${c.label}: ${fmt(c.raw)} (${pct(c.raw / R.total)})` } } } },
      });
    } else { S.graficas.tipos.data.datasets[0].data = vals; S.graficas.tipos.update(); }

    // Barras por raza
    const razas = Object.keys(R.por_raza), vr = razas.map((r) => R.por_raza[r]);
    const colores = razas.map((r) => COLOR_TIPO[E.razas[r]?.tipo] || "#94a3b8");
    if (!S.graficas.razas) {
      S.graficas.razas = new Chart($("#graf-razas"), {
        type: "bar",
        data: { labels: razas, datasets: [{ data: vr, backgroundColor: colores, borderRadius: 5, maxBarThickness: 22 }] },
        options: { ...opcionesBase(), indexAxis: "x",
          scales: { x: { ticks: { autoSkip: false, maxRotation: 60, minRotation: 45, font: { size: 10.5 } }, grid: { display: false } },
            y: { ticks: { callback: (v) => fmt(v) }, grid: { color: "#eef2f8" } } },
          plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => ` ${fmt(c.raw)} registros · ${E.razas[c.label].tipo}` } } } },
      });
    } else { S.graficas.razas.data.datasets[0].data = vr; S.graficas.razas.update(); }

    // Tablas de medias y descriptiva
    $("#tabla-medias tbody").innerHTML = tipos.map((t) => {
      const m = R.medias_por_tipo[t] || {};
      return `<tr><td><span class="etq" style="--c:${COLOR_TIPO[t]}">${t}</span></td><td>${fmt(m.peso_kg ?? 0, 2)}</td><td>${fmt(m.altura_cm ?? 0, 1)}</td><td>${fmt(m.edad_anios ?? 0, 1)}</td></tr>`;
    }).join("");
    $("#tabla-desc tbody").innerHTML = E.numericas.map((c) => {
      const s = R.numericas[c] || {};
      return `<tr><td><b>${NOMBRE_RASGO[c]}</b></td><td>${fmt(s.min ?? 0, 2)}</td><td>${fmt(s.media ?? 0, 2)}</td><td>${fmt(s.mediana ?? 0, 2)}</td><td>${fmt(s.max ?? 0, 2)}</td></tr>`;
    }).join("") + E.categoricas.map((c) => {
      const d = R.categoricas[c] || {}, top = Object.entries(d).sort((a, b) => b[1] - a[1])[0];
      return `<tr><td><b>${NOMBRE_RASGO[c]}</b></td><td colspan="4">${Object.keys(d).length} categorías · moda: <b>${esc(top?.[0] ?? "—")}</b></td></tr>`;
    }).join("");
  }

  // =====================================================================
  //  TABLA PAGINADA
  // =====================================================================
  function sincronizarFiltros() {
    $("#f-tipo").value = S.tabla.tipo;
    llenarFiltroRazas();
    $("#f-raza").value = S.tabla.raza;
    $("#f-buscar").value = S.tabla.buscar;
  }
  function llenarFiltroRazas() {
    const t = $("#f-tipo").value;
    const razas = t ? S.esquema.razas_por_tipo[t] : Object.keys(S.esquema.razas);
    $("#f-raza").innerHTML = `<option value="">Todas las razas</option>` + razas.map((r) => `<option>${esc(r)}</option>`).join("");
  }

  function iniciarTabla() {
    $("#f-tipo").innerHTML += S.esquema.tipos.map((t) => `<option>${t}</option>`).join("");
    llenarFiltroRazas();
    $("#f-tipo").addEventListener("change", () => { S.tabla.tipo = $("#f-tipo").value; S.tabla.raza = ""; llenarFiltroRazas(); S.tabla.pagina = 1; cargarTabla(); });
    $("#f-raza").addEventListener("change", () => { S.tabla.raza = $("#f-raza").value; S.tabla.pagina = 1; cargarTabla(); });
    $("#f-por").addEventListener("change", () => { S.tabla.por = +$("#f-por").value; S.tabla.pagina = 1; cargarTabla(); });
    let espera;
    $("#f-buscar").addEventListener("input", () => { clearTimeout(espera); espera = setTimeout(() => { S.tabla.buscar = $("#f-buscar").value; S.tabla.pagina = 1; cargarTabla(); }, 350); });
    $("#pag-prev").addEventListener("click", () => { S.tabla.pagina--; cargarTabla(); });
    $("#pag-next").addEventListener("click", () => { S.tabla.pagina++; cargarTabla(); });
    $$("#tabla-datos th[data-orden]").forEach((th) => th.addEventListener("click", () => {
      const c = th.dataset.orden;
      S.tabla.desc = S.tabla.orden === c ? !S.tabla.desc : false;
      S.tabla.orden = c; S.tabla.pagina = 1; cargarTabla();
    }));
  }

  async function cargarTabla() {
    const T = S.tabla;
    const q = new URLSearchParams({ pagina: T.pagina, por_pagina: T.por, tipo: T.tipo, raza: T.raza, buscar: T.buscar, orden: T.orden, desc: T.desc ? 1 : 0 });
    const r = await api("/api/dataset/registros?" + q);
    if (!r.ok) return;
    const d = r.datos;
    T.pagina = d.pagina;
    $("#tabla-total").textContent = fmt(d.total) + (T.tipo || T.raza || T.buscar ? " filtrados" : " registros");
    $("#tabla-datos tbody").innerHTML = d.filas.length ? d.filas.map((f) => `
      <tr class="${f.id === T.resaltar ? "nueva" : ""}"><td>${fmt(f.id)}</td><td>${f.peso_kg}</td><td>${f.altura_cm}</td><td>${f.edad_anios}</td>
      <td>${esc(f.nivel_actividad)}</td><td>${esc(f.tipo_pelaje)}</td><td><span class="etq" style="--c:${COLOR_TIPO[f.tipo]}">${esc(f.tipo)}</span></td><td>${esc(f.raza)}</td></tr>`).join("")
      : `<tr class="fila-vacia"><td colspan="8">Sin registros con esos filtros.</td></tr>`;
    const ini = d.total ? (d.pagina - 1) * d.por_pagina + 1 : 0;
    $("#pag-info").textContent = `Mostrando ${fmt(ini)}–${fmt(Math.min(d.pagina * d.por_pagina, d.total))} de ${fmt(d.total)}`;
    $("#pag-num").textContent = `${fmt(d.pagina)} / ${fmt(d.paginas)}`;
    $("#pag-prev").disabled = d.pagina <= 1;
    $("#pag-next").disabled = d.pagina >= d.paginas;
    $$("#tabla-datos th").forEach((th) => th.classList.remove("orden-asc", "orden-desc"));
    $(`#tabla-datos th[data-orden="${T.orden}"]`)?.classList.add(T.desc ? "orden-desc" : "orden-asc");
    T.resaltar = null;
  }

  // =====================================================================
  //  CARGA DE CSV
  // =====================================================================
  function iniciarCargaCSV() {
    const dz = $("#dropzone"), input = $("#in-csv"), btn = $("#btn-subir");
    const elegir = (archivo) => {
      if (!archivo) return;
      if (!archivo.name.toLowerCase().endsWith(".csv")) { toast("Archivo no válido", " Solo se aceptan archivos .csv", "error"); return; }
      const dt = new DataTransfer(); dt.items.add(archivo); input.files = dt.files;
      $("#drop-nombre").textContent = `${archivo.name} · ${(archivo.size / 1048576).toFixed(2)} MB`;
      dz.classList.add("listo"); btn.disabled = false;
    };
    input.addEventListener("change", () => elegir(input.files[0]));
    ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("sobre"); }));
    ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("sobre"); }));
    dz.addEventListener("drop", (e) => elegir(e.dataTransfer.files[0]));

    $("#form-csv").addEventListener("submit", (e) => {
      e.preventDefault();
      const archivo = input.files[0];
      if (!archivo) return;
      const modo = $('#form-csv input[name="modo"]:checked').value;
      if (modo === "reemplazar" && !confirmarReemplazo()) return;
      const fd = new FormData(); fd.append("archivo", archivo); fd.append("modo", modo);
      const xhr = new XMLHttpRequest();
      const prog = $("#progreso"), barra = $("#progreso-barra"), txt = $("#progreso-txt");
      prog.hidden = false; barra.style.width = "0"; cargando([btn], true);
      xhr.upload.onprogress = (ev) => {
        if (!ev.lengthComputable) return;
        const p = Math.round((ev.loaded / ev.total) * 100);
        barra.style.width = p + "%"; txt.textContent = p < 100 ? `Subiendo ${p} %` : "Validando en el servidor…";
      };
      xhr.onload = async () => {
        cargando([btn], false);
        let d; try { d = JSON.parse(xhr.responseText); } catch { d = { estado: "invalido", mensaje: "Respuesta inválida del servidor" }; }
        barra.style.width = "100%"; txt.textContent = d.estado === "invalido" ? "Archivo rechazado" : "Carga completada";
        setTimeout(() => (prog.hidden = true), 2500);
        pintarInforme(d);
        if (d.filas_validas > 0) {
          toast("CSV cargado", ` ${fmt(d.filas_validas)} filas válidas ${d.modo === "reemplazar" ? "reemplazaron el dataset" : "añadidas"}.` + (d.filas_rechazadas ? ` ${fmt(d.filas_rechazadas)} rechazadas.` : ""), d.filas_rechazadas ? "aviso" : "ok", 5500);
          S.tabla.pagina = 1; S.tabla.orden = "id"; S.tabla.desc = true;
          await refrescarResumen(); cargarTabla();
        } else toast("CSV rechazado", " " + (d.mensaje || ""), "error", 6000);
      };
      xhr.onerror = () => { cargando([btn], false); prog.hidden = true; toast("Error de red", " No se pudo subir el archivo.", "error"); };
      xhr.open("POST", "/api/dataset/cargar"); xhr.send(fd);
    });
  }
  function confirmarReemplazo() {
    // Sin diálogos nativos: se exige un segundo clic en 5 s.
    const btn = $("#btn-subir");
    if (btn.dataset.confirmar === "1") { btn.dataset.confirmar = ""; return true; }
    btn.dataset.confirmar = "1";
    toast("¿Reemplazar el dataset completo?", " Presiona otra vez «Validar y cargar» para confirmar.", "aviso", 5000);
    setTimeout(() => (btn.dataset.confirmar = ""), 5000);
    return false;
  }

  const REGLAS_VALIDACION = [
    "Columnas obligatorias presentes",
    "Sin valores faltantes",
    "Tipos de dato numéricos (edad entera)",
    "Rangos físicos permitidos",
    "Categorías dentro del catálogo",
    "Coherencia raza → tipo y tipo → pelaje",
  ];

  function pintarInforme(d) {
    const est = $("#inf-estado");
    const mapa = { ok: ["Válido", "verde"], con_errores: ["Con errores", "ambar"], invalido: ["Rechazado", "rojo"] };
    const [t, c] = mapa[d.estado] || ["—", "gris"];
    est.className = "pill " + c; est.textContent = d.archivo ? `${t} · ${d.archivo}` : t;
    $("#inf-leidas").textContent = fmt(d.filas_leidas || 0);
    $("#inf-validas").textContent = fmt(d.filas_validas || 0);
    $("#inf-rech").textContent = fmt(d.filas_rechazadas || 0);
    $("#inf-norm").textContent = fmt(d.normalizados || 0);
    const resumen = d.resumen_errores || {};
    $("#inf-resumen").innerHTML = (d.columnas_faltantes?.length ? `<span>Faltan columnas: ${esc(d.columnas_faltantes.join(", "))}</span>` : "")
      + Object.entries(resumen).map(([m, n]) => `<span>${esc(m)}: ${fmt(n)}</span>`).join("");
    const errs = d.errores || [];
    $("#tabla-errores tbody").innerHTML = errs.length ? errs.map((e) => `<tr><td>${e.fila}</td><td>${esc(e.columna)}</td><td>${esc(e.valor) || "<i>(vacío)</i>"}</td><td>${esc(e.motivo)}</td></tr>`).join("")
      : `<tr class="fila-vacia"><td colspan="4">${d.estado === "invalido" ? esc(d.mensaje) : "Sin errores: todas las filas pasaron la validación."}</td></tr>`;
    const motivos = Object.keys(resumen).join(" ").toLowerCase();
    if (!d.estado) {
      $("#tabla-errores tbody").innerHTML = `<tr class="fila-vacia"><td colspan="4">Aún no se ha cargado ningún archivo. Elige un CSV para validarlo.</td></tr>`;
      $("#inf-checks").innerHTML = REGLAS_VALIDACION.map((r) => `<li class="pend">${icono("shield-check")}<span>${r}</span></li>`).join("");
      return;
    }
    const falla = [d.columnas_faltantes?.length, /faltante/.test(motivos), /numérico|entero/.test(motivos), /rango/.test(motivos), /catálogo/.test(motivos), /corresponde|incompatible/.test(motivos)];
    $("#inf-checks").innerHTML = REGLAS_VALIDACION.map((r, i) => `<li class="${falla[i] ? "mal" : ""}">${icono(falla[i] ? "circle-x" : "circle-check")}<span>${r}</span></li>`).join("");
  }

  // =====================================================================
  //  DICCIONARIO DE DATOS
  // =====================================================================
  async function pintarDiccionario() {
    const E = S.esquema, R = S.resumen;
    const rol = (c) => (c === "id" ? "Identificador" : E.etiquetas.includes(c) ? "Etiqueta" : "Predictora");
    const tipo = (c) => c === "id" || c === "edad_anios" ? "Entero" : E.numericas.includes(c) ? "Decimal" : c === "nivel_actividad" ? "Categórica ordinal" : "Categórica";
    const dominio = (c) => {
      if (c === "id") return `1 – ${fmt(R.total)}`;
      if (E.numericas.includes(c)) return `${E.rangos[c][0]} – ${E.rangos[c][1]}`;
      if (c === "nivel_actividad") return E.niveles_actividad.join(" < ");
      if (c === "tipo_pelaje") return E.tipos_pelaje.join(", ");
      if (c === "tipo") return E.tipos.join(", ");
      return `${Object.keys(E.razas).length} razas (5 por tipo)`;
    };
    $("#tabla-dicc tbody").innerHTML = Object.keys(E.descripciones).map((c) =>
      `<tr><td><code>${c}</code></td><td>${rol(c)}</td><td>${tipo(c)}</td><td>${esc(dominio(c))}</td><td>${esc(E.descripciones[c])}</td></tr>`).join("");

    const r = await api("/api/dataset/diccionario");
    const q = r.datos?.calidad;
    const items = q ? [
      [true, `${fmt(q.filas_leidas)} registros generados; ${fmt(q.filas_validas)} válidos y ${fmt(q.filas_rechazadas)} rechazados`],
      [!Object.values(q.faltantes_por_columna || {}).some(Boolean), "0 valores faltantes en las 7 columnas"],
      [true, "5 predictoras: 3 numéricas + 2 categóricas"],
      [true, "IDs únicos y consecutivos"],
      [true, "Rangos físicos verificados por variable"],
      [true, "Coherencia raza → tipo y tipo → pelaje"],
      [true, "Generación reproducible (semilla fija)"],
    ] : [[false, "Ejecuta data/generar_dataset.py para crear el reporte de calidad"]];
    $("#calidad-checks").innerHTML = items.map(([ok, t]) => `<li class="${ok ? "" : "mal"}">${icono(ok ? "circle-check" : "circle-x")}<span>${esc(t)}</span></li>`).join("");
  }

  // =====================================================================
  //  ACCIONES DE MODELOS (Integrantes 2 y 3) — contrato en docs/CONTRATO_API.md
  // =====================================================================
  const ACCIONES = {
    entrenar: { url: "/api/entrenar", cuerpo: () => ({}), pintar: pintarEntrenamiento, ok: "Modelos reentrenados", vista: "entrenamiento" },
    clasificar: { url: "/api/clasificar", cuerpo: datosClasificar, pintar: pintarPrediccion, ok: "Clasificación lista" },
    agrupar: { url: "/api/agrupar", cuerpo: () => ({ k: +$("#in-k").value }), pintar: pintarKmeans, ok: "Agrupación lista" },
    asociar: { url: "/api/asociar", cuerpo: () => ({  min_support: +$("#in-soporte").value,   min_confidence: +$("#in-confianza").value  }), pintar: pintarApriori,  ok: "Reglas generadas"},
  };

  function datosClasificar() {
    const d = leerFormulario();
    const err = validarCliente(d, false);
    if (Object.keys(err).length) { mostrarErrores(err); throw new Error("Corrige los rasgos antes de clasificar."); }
    const out = {};
    S.esquema.predictoras.forEach((c) => (out[c] = S.esquema.numericas.includes(c) ? Number(String(d[c]).replace(",", ".")) : d[c]));
    return out;
  }

  async function ejecutar(accion) {
    const A = ACCIONES[accion];
    let cuerpo;
    try { cuerpo = A.cuerpo(); } catch (e) { toast("Revisa los datos", " " + e.message, "error"); return; }
    const botones = $$(`[data-accion="${accion}"]`);
    cargando(botones, true);
    const r = await api(A.url, { json: cuerpo });
    cargando(botones, false);
    if (r.status === 501 || r.datos?.pendiente) {
      toast("Módulo pendiente de integrar", ` ${r.datos?.mensaje || ""}`, "aviso", 5000);
      return;
    }
    if (!r.ok) { toast("Error", " " + (r.datos?.mensaje || r.datos?.error || `Código ${r.status}`), "error"); return; }
    A.pintar(r.datos);
    toast(A.ok, "", "ok", 2500);
    if (accion === "entrenar") refrescarResumen();
  }

  function pintarEntrenamiento(d) {
      // Guardar toda la respuesta para poder alternar entre Tipo y Raza.
      S.resultadosEntrenamiento = d;

      if (d.division) {
          S.division = d.division;
          $("#n-train").textContent = fmt(d.division.train);
          $("#n-val").textContent = fmt(d.division.val);
          $("#n-test").textContent = fmt(d.division.test);
      }

      // Guardar las dos matrices de confusión.
      if (d.matrices) {
          S.matrices = d.matrices;
      }

      // Mostrar los resultados correspondientes al objetivo seleccionado.
      pintarResultadosNivel();

      // Actualizar el estado general del entrenamiento.
      $("#train-nota").innerHTML =
          `Modelos entrenados para clasificación por tipo y raza. ` +
          `La selección se realiza mediante <b>validación</b> y la evaluación final utiliza <b>prueba</b> (${fmt(d.division?.test || 0)} registros).` +
          (d.entrenado_en
              ? ` · ${esc(new Date(d.entrenado_en).toLocaleString("es-MX"))}`
              : "");
  }

  function pintarResultadosNivel() {
      const d = S.resultadosEntrenamiento;
      if (!d) return;

      const nivel = S.nivelMatriz;
      const esRaza = nivel === "raza";

      const modelos = esRaza
          ? (d.modelos_raza || [])
          : (d.modelos || []);

      const mejor = esRaza
          ? d.mejor_raza
          : d.mejor;

      const metricas = esRaza
          ? d.test_raza
          : d.test;

      // Actualizar la tabla de algoritmos.
      const tbody = $("#tabla-algos tbody");

      tbody.innerHTML = modelos.map((m) => `
          <tr class="${m.nombre === mejor ? "mejor" : ""}"
              data-modelo="${esc(m.nombre)}">
              <td>${esc(m.nombre)}</td>
              <td>${f3(m.accuracy)}</td>
              <td>${f3(m.error_total)}</td>
              <td>${f3(m.fpr_macro)}</td>
              <td>${f3(m.fnr_macro)}</td>
              <td>${f3(m.f1_macro)}</td>
          </tr>
      `).join("");

      // Identificar el objetivo de la comparación.
      $("#tabla-algo-nota").textContent = esRaza
          ? "(clasificación de razas · métricas en prueba)"
          : "(clasificación de tipos · métricas en prueba)";

      // Mostrar el mejor modelo seleccionado.
      $("#train-nota").innerHTML =
          `Objetivo: <b>${esRaza ? "Clasificación por raza" : "Clasificación por tipo"}</b>. ` +
          `Mejor modelo según <b>validación</b>: <b>${esc(mejor || "—")}</b>.`;

      // Actualizar el nombre del modelo en la tarjeta de la matriz.
      $("#matriz-modelo").textContent = mejor
          ? `(${mejor})`
          : "";

      // Actualizar las métricas del objetivo seleccionado.
      if (metricas) {
          pintarMetricas(metricas, mejor, null);
      }

      // Actualizar la matriz correspondiente.
      pintarMatriz();
  }

  function pintarMetricas(t, modelo, texto) {
    $("#metricas-vacio").hidden = true;
    const etq = ["Accuracy", "F1 (macro)", "FPR (macro)", "FNR (macro)", "Error total"];
    const val = [t.accuracy, t.f1_macro, t.fpr_macro, t.fnr_macro, t.error_total].map(Number);
    const col = ["#1f6feb", "#7c3aed", "#f59e0b", "#ef4444", "#64748b"];
    const etiquetasValor = {
      id: "valores", afterDatasetsDraw(ch) {
        const { ctx } = ch; ctx.save(); ctx.font = "700 12px " + Chart.defaults.font.family; ctx.fillStyle = "#0f1f3a"; ctx.textAlign = "center";
        ch.getDatasetMeta(0).data.forEach((b, i) => ctx.fillText(val[i].toFixed(3), b.x, b.y - 6)); ctx.restore();
      },
    };
    S.graficas.metricas?.destroy();
    S.graficas.metricas = new Chart($("#graf-metricas"), {
      type: "bar", data: { labels: etq, datasets: [{ data: val, backgroundColor: col, borderRadius: 6, maxBarThickness: 64 }] },
      options: { ...opcionesBase(), layout: { padding: { top: 22 } }, scales: { y: { min: 0, max: 1, ticks: { stepSize: 0.2 }, grid: { color: "#eef2f8" } }, x: { grid: { display: false } } } },
      plugins: [etiquetasValor],
    });
    $("#metricas-texto").innerHTML = texto ? esc(texto)
      : `En el conjunto de <b>prueba</b>, ${esc(modelo || "el mejor modelo")} clasifica correctamente el <b>${pct(t.accuracy)}</b> de las mascotas (error total ${pct(t.error_total)}). `
      + `En promedio por clase, el <b>${pct(t.fpr_macro)}</b> de los casos negativos se marcan como positivos (FPR) y el <b>${pct(t.fnr_macro)}</b> de los positivos no se detectan (FNR).`;
  }

  function pintarMatriz() {
    const m = S.matrices?.[S.nivelMatriz];

    if (!m || !Array.isArray(m.valores) || !Array.isArray(m.etiquetas)) {
      $("#matriz-wrap").innerHTML = `
          <div class="vacio-grande">
              ${icono("table")}
              <p>No hay una matriz disponible para este objetivo. Reentrena los modelos.</p>
          </div>`;
      return;
    }

    const max = Math.max(0, ...m.valores.flat());
    const comp = m.etiquetas.length > 6;
    const corta = (s) => (comp ? s.split(" ")[0].slice(0, 9) : s);

    let h = `<table class="matriz ${comp ? "compacta" : ""}">
        <tr>
            <th></th>
            <th></th>
            ${m.etiquetas.map((e) =>
                `<th title="${esc(e)}">${esc(corta(e))}</th>`
            ).join("")}
        </tr>`;

    m.valores.forEach((fila, i) => {
        h += `<tr>
            ${i === 0
                ? `<th rowspan="${m.etiquetas.length}" class="eje-y">Clase real</th>`
                : ""}
            <th style="text-align:right"
                title="${esc(m.etiquetas[i])}">
                ${esc(corta(m.etiquetas[i]))}
            </th>`;

        fila.forEach((v, j) => {
            const k = max ? v / max : 0;

            const fondo = i === j
                ? `rgba(21,94,214,${0.35 + 0.65 * k})`
                : `rgba(31,111,235,${Math.min(0.35, 0.05 + k * 2)})`;

            h += `<td
                style="background:${fondo};color:${i === j && k > 0.3 ? "#fff" : "#0f1f3a"}"
                title="Real: ${esc(m.etiquetas[i])} · Predicha: ${esc(m.etiquetas[j])}">
                ${fmt(v)}
            </td>`;
        });

        h += "</tr>";
    });

    h += `<tr>
        <td></td>
        <td></td>
        <td colspan="${m.etiquetas.length}" class="eje-x" style="text-align:center">
            Clase predicha
        </td>
    </tr></table>`;

    $("#matriz-wrap").innerHTML = h;
  }

  async function pintarPrediccion(d) {
    S.ultimaPrediccion = d;
    $("#pred-tipo").textContent = d.tipo;
    $("#pred-raza").textContent = d.raza;
    $("#pred-modelo").innerHTML = `Modelo: <b>${esc(d.modelo || "—")}</b> · confianza en la raza ${pct(d.prob_raza?.[d.raza] ?? 0)}`;
    const box = $("#pred-resultado"); box.classList.remove("activo"); void box.offsetWidth; box.classList.add("activo");
    const barras = (obj, colorDe, n) => Object.entries(obj || {}).sort((a, b) => b[1] - a[1]).slice(0, n)
      .map(([k, v]) => `<li><span title="${esc(k)}">${esc(k)}</span><span class="pista"><span class="relleno" style="background:${colorDe(k)}" data-w="${(v * 100).toFixed(1)}"></span></span><b>${(v * 100).toFixed(1)} %</b></li>`).join("");
    $("#prob-tipo").innerHTML = barras(d.prob_tipo, (k) => COLOR_TIPO[k] || "#1f6feb", 4);
    $("#prob-raza").innerHTML = barras(d.prob_raza, (k) => COLOR_TIPO[S.esquema.razas[k]?.tipo] || "#7c3aed", 3) || `<li class="vacio">—</li>`;
    requestAnimationFrame(() => $$(".relleno[data-w]").forEach((el) => (el.style.width = el.dataset.w + "%")));
    $("#pred-chip").textContent = "Predicha"; $("#pred-chip").classList.add("predicha");
    $("#pred-cap-raza").textContent = `${d.tipo} · ${d.raza}`;
    ponerImagen($("#pred-img"), d.raza);   // imagen vinculada a la ETIQUETA PREDICHA
  }

  function pintarProbVacias() {
    $("#prob-tipo").innerHTML = S.esquema.tipos.map((t) => `<li><span>${t}</span><span class="pista"><span class="relleno" style="background:${COLOR_TIPO[t]}"></span></span><b>—</b></li>`).join("");
  }

  function pintarKmeans(d) {
    $("#kmeans-vacio").hidden = true;
    $("#k-label").textContent = d.k;
    const ex = d.ejes?.x || "peso_kg", ey = d.ejes?.y || "altura_cm";
    const grupos = [...Array(d.k).keys()].map((g) => {
      const c = (d.centroides || []).find((x) => x.grupo === g) || {};
      return {
        label: c.etiqueta || `Grupo ${g + 1}`, data: (d.puntos || []).filter((p) => p[2] === g).map((p) => ({ x: p[0], y: p[1] })),
        backgroundColor: COLOR_GRUPO[g % 8] + "99", pointRadius: 2.6, pointHoverRadius: 5,
      };
    });
    grupos.push({
      label: "Centroides", data: (d.centroides || []).map((c) => ({ x: c.x, y: c.y })), pointStyle: "crossRot",
      pointRadius: 11, pointBorderWidth: 3.5, borderColor: "#0f1f3a", backgroundColor: "#0f1f3a",
    });
    S.graficas.kmeans?.destroy();
    S.graficas.kmeans = new Chart($("#graf-kmeans"), {
      type: "scatter", data: { datasets: grupos },
      options: { ...opcionesBase(), animation: { duration: 500 },
        plugins: { legend: { display: true, position: "right", labels: { usePointStyle: true, boxWidth: 8, padding: 10 } } },
        scales: { x: { title: { display: true, text: NOMBRE_RASGO[ex] || ex }, grid: { color: "#eef2f8" } },
          y: { title: { display: true, text: NOMBRE_RASGO[ey] || ey }, grid: { color: "#eef2f8" } } } },
    });
    $("#kmeans-texto").innerHTML = (d.interpretacion || []).map((t) => `<li>${esc(t)}</li>`).join("");
  }

  function pintarApriori(d) {
    const reglas = d.reglas || [];
    $("#tabla-reglas tbody").innerHTML = reglas.length ? reglas.map((r) =>
      `<tr><td>${esc((r.antecedente || []).join(" ∧ "))} <b>→ ${esc((r.consecuente || []).join(" ∧ "))}</b></td><td>${f3(r.soporte)}</td><td>${f3(r.confianza)}</td><td><b>${Number(r.lift).toFixed(2)}</b></td></tr>`).join("")
      : `<tr class="fila-vacia"><td colspan="4">No hay reglas con esos umbrales. Baja el soporte o la confianza.</td></tr>`;
    $("#apriori-texto").innerHTML = d.interpretacion ? esc(d.interpretacion)
      : `${fmt(reglas.length)} reglas sobre ${fmt(d.n_transacciones || 0)} transacciones. Un <b>lift &gt; 1</b> indica que el antecedente aumenta la probabilidad del consecuente.`;
  }

  function iniciarAcciones() {
    document.addEventListener("click", (e) => {
      const b = e.target.closest("[data-accion]");
      if (b) { e.preventDefault(); ejecutar(b.dataset.accion); }
    });
    $("#in-k").addEventListener("input", () => ($("#k-label").textContent = $("#in-k").value));
    $("#in-soporte").addEventListener("input", () => ($("#out-sop").textContent = Number($("#in-soporte").value).toFixed(2)));
    $("#in-confianza").addEventListener("input", () => ($("#out-conf").textContent = Number($("#in-confianza").value).toFixed(2)));

    // --- Selector de nivel de la matriz (Tipo / Raza) ---
    // Al cambiar de nivel se debe repintar TODO: tabla de algoritmos, métricas y matriz.
    $$("#matriz-nivel button").forEach((b) => {
        b.addEventListener("click", () => {
            $$("#matriz-nivel button").forEach((x) => {
                x.classList.toggle("activo", x === b);
            });

            S.nivelMatriz = b.dataset.nivel;

            // Sincronizar el selector de objetivos del entrenamiento.
            const selector = $("#entrenamiento-objetivo");
            if (selector) selector.value = S.nivelMatriz;

            pintarResultadosNivel();
        });
    });

    // --- Selector de objetivo del entrenamiento ---
    // Mantiene sincronizados ambos controles (Tipo/Raza).
    const selectorObjetivo = $("#entrenamiento-objetivo");

    if (selectorObjetivo) {
        selectorObjetivo.addEventListener("change", () => {
            S.nivelMatriz = selectorObjetivo.value;

            $$("#matriz-nivel button").forEach((b) => {
                b.classList.toggle("activo", b.dataset.nivel === S.nivelMatriz);
            });

            pintarResultadosNivel();
        });
    }

    // al cambiar los rasgos, la predicción anterior deja de ser válida
    F.form().addEventListener("input", () => { if (S.ultimaPrediccion) { S.ultimaPrediccion = null; $("#pred-resultado").classList.remove("activo"); vistaPrevia(); } });
    F.form().addEventListener("change", (e) => { if (S.ultimaPrediccion && ["tipo", "raza"].includes(e.target.name)) { S.ultimaPrediccion = null; vistaPrevia(); } });
  }

  // =====================================================================
  //  ARRANQUE
  // =====================================================================
  async function iniciar() {
    hidratarIconos();
    const r = await api("/api/esquema");
    if (!r.ok) { toast("No se pudo iniciar", " Verifica que Flask esté ejecutándose.", "error", 10000); return; }
    S.esquema = r.datos;
    hidratarIconos($("#almacen"));
    iniciarNavegacion();
    iniciarSelector();
    iniciarTabla();
    iniciarCargaCSV();
    iniciarAcciones();
    pintarProbVacias();
    pintarInforme({ estado: null, filas_leidas: 0 });
    $("#inf-estado").className = "pill gris"; $("#inf-estado").textContent = "Sin cargas";
    await refrescarResumen();
    pintarDiccionario();
    $$("#topbar-mascotas img").forEach((img) => ponerImagen(img, img.dataset.raza));
    // Exponer funciones de pintado para pruebas e integración
    window.PetIA = { pintarEntrenamiento, pintarPrediccion, pintarKmeans, pintarApriori, refrescarResumen, irA };
  }

  document.addEventListener("DOMContentLoaded", iniciar);
})();