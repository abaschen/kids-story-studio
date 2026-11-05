// Kids Story Studio UI: plain JS, polls the story while a job runs.
const $ = (s) => document.querySelector(s)
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c])
const api = async (path, opts = {}) => {
  const r = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts, body: opts.body && JSON.stringify(opts.body) })
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText)
  return r.json()
}
const file = (s, p) => `/files/${s.id}/${p}?v=${encodeURIComponent(s.updated)}`

let current = null // story
let tab = 'story'
let bookPage = 0
let timer = null

// -- list & health -------------------------------------------------------------------------------------------------
async function refreshList() {
  const list = await api('/api/stories')
  $('#stories').innerHTML = list
    .map((s) => `<li data-id="${esc(s.id)}" class="${current?.id === s.id ? 'on' : ''}">
      ${s.cover ? `<img src="/files/${esc(s.id)}/${esc(s.cover)}" alt="">` : '<span class="blank"></span>'}
      <span>${esc(s.title)}<small>${s.pages ? `${s.pages} pages · ` : ''}${esc(s.created.slice(0, 10))}${s.state === 'running' || s.state === 'queued' ? ' · ⏳' : ''}</small></span></li>`)
    .join('')
}
$('#stories').addEventListener('click', (e) => {
  const li = e.target.closest('li')
  if (li) open(li.dataset.id)
})
async function refreshHealth() {
  try {
    const h = await api('/api/health')
    $('#health').innerHTML = `Story: ${esc(h.story_model)} ${h.ollama ? (h.story_model_installed ? '✅' : '⬇️ on first use') : '❌ ollama down'}<br>
      Pictures: ${esc(h.image_model)}<br>GPU memory used: ${h.vram_used_mb ?? '?'} MB`
  } catch {
    $('#health').textContent = 'server unreachable'
  }
}

// Story model picker: installed models first; the others download on first use.
async function refreshModels() {
  const m = await api('/api/models')
  const opt = (n, label) => `<option value="${esc(n)}" ${n === m.current ? 'selected' : ''}>${esc(label)}</option>`
  const known = m.installed.map((i) => i.name)
  $('#model').innerHTML =
    m.installed.map((i) => opt(i.name, `${i.name} (${i.size_gb} GB)`)).join('') +
    (known.includes(m.current) ? '' : opt(m.current, m.current)) +
    m.suggested.filter((n) => n !== m.current).map((n) => opt(n, `${n} (downloads on first use)`)).join('') +
    '<option value="__other">other…</option>'
}
$('#model').onchange = async (e) => {
  let name = e.target.value
  if (name === '__other') name = (prompt('Ollama model name (e.g. qwen3:14b):') || '').trim()
  try {
    if (name) await api('/api/settings', { method: 'PUT', body: { story_model: name } })
  } catch (err) {
    alert(err.message)
  }
  refreshModels()
  refreshHealth()
}
$('#delete').onclick = async () => {
  if (!confirm(`Delete "${current.title || 'this story'}" with all its pictures and logs? This cannot be undone.`)) return
  try {
    await api(`/api/stories/${current.id}`, { method: 'DELETE' })
  } catch (e) {
    return alert(e.message)
  }
  clearTimeout(timer)
  current = null
  $('#view').hidden = true
  await refreshList()
  const first = document.querySelector('#stories li')
  first ? open(first.dataset.id) : $('#new').click()
}

// -- create --------------------------------------------------------------------------------------------------------
$('#new').onclick = () => {
  current = null
  $('#view').hidden = true
  $('#create').hidden = false
  $('#context').value = localStorage.getItem('story:context') || ''
  $('#language').value = localStorage.getItem('story:language') || 'English'
  refreshList()
}
$('#go').onclick = async () => {
  const body = { context: $('#context').value, prompt: $('#prompt').value, language: $('#language').value, auto: $('#auto').checked }
  localStorage.setItem('story:context', body.context)
  localStorage.setItem('story:language', body.language)
  $('#go').disabled = true
  try {
    const s = await api('/api/stories', { method: 'POST', body })
    $('#prompt').value = ''
    await open(s.id)
  } catch (e) {
    alert(e.message)
  } finally {
    $('#go').disabled = false
  }
}

// -- view ----------------------------------------------------------------------------------------------------------
async function open(id) {
  current = await api(`/api/stories/${id}`)
  $('#create').hidden = true
  $('#view').hidden = false
  bookPage = 0
  tab = current.pages.some((p) => p.image) ? 'book' : current.pages.length ? 'pages' : 'story'
  render(true)
  refreshList()
  poll()
}

const busy = () => ['queued', 'running'].includes(current?.status.state)

function poll() {
  clearTimeout(timer)
  if (!current) return
  timer = setTimeout(async () => {
    const was = current.updated
    const wasBusy = busy()
    current = await api(`/api/stories/${current.id}`)
    if (current.updated !== was) render(false)
    if (wasBusy !== busy()) refreshList()
    poll()
  }, busy() ? 1500 : 6000)
}

function render(full) {
  const s = current
  $('#title').textContent = s.title || 'Writing…'
  const st = s.status
  const el = $('#status')
  el.className = st.state
  el.innerHTML =
    st.state === 'error' ? `⚠️ ${esc(st.error)}` : busy() ? `${esc(st.step)}${st.total ? ` <progress max="${st.total}" value="${st.done}"></progress>` : ''}` : s.pages.some((p) => p.image) ? '✅ book ready' : ''
  document.querySelectorAll('.tabs button').forEach((b) => b.classList.toggle('on', b.dataset.tab === tab))
  document.querySelectorAll('.tab').forEach((t) => (t.hidden = t.id !== `tab-${tab}`))
  document.querySelectorAll('#view .row button').forEach((b) => (b.disabled = busy()))
  if (tab === 'story') renderStory(full)
  // Don't redraw under the cursor: a focused text box keeps what's being typed.
  const typing = document.activeElement?.matches?.('textarea, input')
  if (tab === 'pages' && (full || !typing)) renderPages()
  if (tab === 'book') renderBook()
  if (tab === 'log') renderLog()
}

document.querySelectorAll('.tabs button').forEach((b) =>
  b.addEventListener('click', () => {
    tab = b.dataset.tab
    render(true)
  }),
)

// Story tab
function renderStory(full) {
  const s = current
  const checklist = (s.beats || []).map((b, i) => `${i + 1}. ${b}`).join('\n')
  $('#request').textContent = `Context: ${s.input.context}\n\n${s.input.prompt}\n\nLanguage: ${s.input.language} · seed ${s.seed}${checklist ? `\n\nChecklist the editor uses:\n${checklist}` : ''}`
  if (full || document.activeElement !== $('#story')) $('#story').value = s.story ? `${s.title}\n\n${s.story}` : ''
  $('#to-pages').disabled = busy() || !s.story
  $('#reviews').innerHTML = [...s.reviews]
    .reverse()
    .map(
      (r) => `<div class="review"><span class="${r.passed ? 'pass' : 'revise'}">Round ${r.round + 1}: ${r.passed ? 'passed ✓' : 'needs work'}</span>
      ${r.teaching_moment ? ` · lesson: <i>${esc(r.teaching_moment)}</i>` : ''}
      <ul>${r.beats.map((b) => `<li class="${b.present ? '' : 'no'}" title="${esc(b.evidence || '')}">${b.present ? '✓' : '✗'} ${esc(b.beat)}${b.note ? ` <small>(${esc(b.note)})</small>` : ''}</li>`).join('')}</ul>
      ${r.problems.length ? `<b>Problems</b><ul>${r.problems.map((p) => `<li>${esc(p)}</li>`).join('')}</ul>` : ''}</div>`,
    )
    .join('')
}
const act = async (path, body) => {
  try {
    current = await api(`/api/stories/${current.id}${path}`, { method: 'POST', body })
    render(true)
    poll()
  } catch (e) {
    alert(e.message)
  }
}
$('#save-story').onclick = async () => {
  const [title, ...rest] = $('#story').value.split('\n')
  current = await api(`/api/stories/${current.id}/story`, { method: 'PUT', body: { title: title.trim(), story: rest.join('\n').trim() } })
  render(true)
}
$('#rewrite').onclick = () => confirm('Write a new draft from your request? The current text and pages are replaced.') && act('/run/write')
$('#revise').onclick = () => {
  const feedback = $('#feedback').value.trim()
  if (!feedback) return
  $('#feedback').value = ''
  act('/revise', { feedback })
}
$('#to-pages').onclick = () => {
  tab = 'pages'
  act('/run/plan')
}

// Pages tab
function renderPages() {
  const s = current
  const b = s.bible
  $('#bible').innerHTML = b
    ? `<div class="char style"><b>Art style</b> ${esc(b.style)}</div>` +
      b.characters
        .map((c) => `<div class="char">${s.refs[c.id] ? `<img src="${file(s, s.refs[c.id])}" alt="">` : ''}<b>${esc(c.name)}</b> · ${esc(c.role)}, ${esc(c.age)}<br>${esc(c.look)}</div>`)
        .join('')
    : busy() ? '' : '<p>No pages yet. Approve the story to split it into pages.</p>'
  $('#pages').innerHTML = s.pages
    .map(
      (p) => `<div class="page" data-n="${p.n}">
        <div class="n">${p.n}</div>
        <textarea class="text" rows="5" title="Text read aloud">${esc(p.text)}</textarea>
        <textarea class="ill" rows="5" title="Illustration">${esc(p.illustration)}</textarea>
        <div class="pic">${p.image ? `<img src="${file(s, p.image)}" alt="">` : '<img alt="">'}
          <div class="meta">${esc(p.setting)} · ${p.characters.map(esc).join(', ')}</div>
          <button class="redo" ${busy() ? 'disabled' : ''}>${p.image ? '🎲 Paint again' : '🎨 Paint'}</button></div>
      </div>`,
    )
    .join('')
  $('#render').disabled = busy() || !s.pages.length
}
$('#pages').addEventListener('click', (e) => {
  const btn = e.target.closest('.redo')
  if (!btn) return
  const n = Number(btn.closest('.page').dataset.n)
  const had = current.pages.find((p) => p.n === n)?.image
  act('/run/render', { pages: [n], new_seed: !!had })
})
$('#save-pages').onclick = async () => {
  const pages = current.pages.map((p) => {
    const el = document.querySelector(`.page[data-n="${p.n}"]`)
    return { ...p, text: el.querySelector('.text').value, illustration: el.querySelector('.ill').value }
  })
  current = await api(`/api/stories/${current.id}/pages`, { method: 'PUT', body: pages })
  render(true)
}
$('#replan').onclick = () => confirm('Split the story into pages again? Page edits and pictures are replaced.') && act('/run/plan')
$('#render').onclick = () => act('/run/render', {})

// Book tab
function renderBook() {
  const pages = current.pages
  const ready = pages.some((p) => p.image)
  for (const [id, layout] of [['#pdf-pages', 'pages'], ['#pdf-booklet', 'booklet']]) {
    $(id).href = `/api/stories/${current.id}/book.pdf?layout=${layout}`
    $(id).classList.toggle('off', !ready)
  }
  if (!pages.length) {
    $('#page-text').textContent = 'No pages yet.'
    $('#page-img').removeAttribute('src')
    return
  }
  bookPage = Math.max(0, Math.min(bookPage, pages.length - 1))
  const p = pages[bookPage]
  if (p.image) $('#page-img').src = file(current, p.image)
  else $('#page-img').removeAttribute('src')
  $('#page-text').textContent = p.text
  $('#page-num').textContent = `${bookPage + 1} / ${pages.length}`
}
$('#prev').onclick = () => { bookPage--; renderBook() }
$('#next').onclick = () => { bookPage++; renderBook() }
document.addEventListener('keydown', (e) => {
  if (tab !== 'book') return
  if (e.key === 'ArrowLeft') $('#prev').click()
  if (e.key === 'ArrowRight') $('#next').click()
})

// Log tab
async function renderLog() {
  const calls = await api(`/api/stories/${current.id}/calls`)
  $('#log').innerHTML =
    '<tr><th>When</th><th>Step</th><th>Model</th><th>Parameters</th><th>Time</th><th>Details</th></tr>' +
    calls
      .map((c) => {
        const ver = c.kind === 'llm' ? `${c.model} · ${String(c.digest).slice(0, 12)}<br><small>${esc(c.runtime)}</small>` : c.kind.startsWith('image') ? `${c.model}@${String(c.revision).slice(0, 10)}<br><small>${esc(c.runtime)}</small>` : 'hand edit'
        const out = c.kind === 'image' ? `<a href="${file(current, c.output)}" target="_blank">${esc(c.output)}</a>` : ''
        return `<tr><td>${esc(c.at.slice(11, 19))}</td><td>${esc(c.stage)}</td><td>${ver}</td>
        <td><pre>${esc(JSON.stringify(c.params ?? {}, null, 1))}</pre></td><td>${c.duration_s ?? ''}s</td>
        <td>${out}<details><summary>request / response</summary><pre>${esc(JSON.stringify(c.request ?? c.before, null, 1))}</pre><pre>${esc(c.response ?? JSON.stringify(c.after ?? '', null, 1))}</pre></details></td></tr>`
      })
      .join('')
}

refreshModels()
refreshHealth()
setInterval(refreshHealth, 10000)
refreshList().then(async () => {
  const first = document.querySelector('#stories li')
  if (first) open(first.dataset.id)
  else $('#new').click()
})
