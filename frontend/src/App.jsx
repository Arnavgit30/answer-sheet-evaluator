import { useMemo, useState } from 'react'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  })
  const data = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(data.detail || 'The server could not complete the request.')
  return data
}

const marks = (value) => (value === null || value === undefined ? '—' : Number(value).toFixed(2).replace(/\.00$/, ''))
const score = (value) => (value === null || value === undefined ? '—' : `${Number(value).toFixed(1)}%`)

export default function App() {
  const [configPath, setConfigPath] = useState('configs/dataset_config.yaml')
  const [validation, setValidation] = useState(null)
  const [studentId, setStudentId] = useState('')
  const [evaluation, setEvaluation] = useState(null)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const selectedStudent = useMemo(
    () => validation?.manifest?.student_documents?.find((student) => student.document_id === studentId),
    [validation, studentId],
  )

  const validateDataset = async () => {
    setLoading(true)
    setError('')
    try {
      const data = await request('/api/v1/datasets/validate', {
        method: 'POST',
        body: JSON.stringify({ config_path: configPath }),
      })
      setValidation(data)
      setResults(null)
      setEvaluation(null)
      if (data.valid) setStudentId(data.manifest.student_documents[0]?.document_id || '')
      else setError(data.errors.join(' '))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const fetchResults = async (evaluationId) => {
    const data = await request(`/api/v1/evaluations/${evaluationId}/results`)
    setResults(data)
  }

  const pollEvaluation = async (evaluationId) => {
    const data = await request(`/api/v1/evaluations/${evaluationId}`)
    setEvaluation(data)
    if (data.status === 'completed') return fetchResults(evaluationId)
    if (data.status === 'failed') throw new Error(data.error || 'Evaluation failed.')
    window.setTimeout(() => pollEvaluation(evaluationId).catch((err) => setError(err.message)), 1200)
  }

  const startEvaluation = async () => {
    if (!validation?.valid || !studentId) return setError('Validate a dataset and select a student answer sheet first.')
    setLoading(true)
    setError('')
    setResults(null)
    try {
      const data = await request('/api/v1/evaluations', {
        method: 'POST',
        body: JSON.stringify({ config_path: configPath, student_document_id: studentId }),
      })
      setEvaluation(data)
      if (data.status === 'completed') await fetchResults(data.evaluation_id)
      else if (data.status === 'failed') throw new Error(data.error || 'Evaluation failed.')
      else await pollEvaluation(data.evaluation_id)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const questions = results?.grading?.questions || []
  const benchmark = results?.benchmark

  return (
    <main className="mx-auto min-h-screen max-w-6xl px-4 py-8 sm:px-6">
      <header className="mb-8">
        <p className="text-sm font-semibold uppercase tracking-[0.2em] text-indigo-600">Local solution demo</p>
        <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-950">Answer Sheet Evaluator</h1>
        <p className="mt-2 max-w-3xl text-slate-600">Validate a configured local dataset, evaluate one student answer sheet, and inspect transparent baseline similarity results.</p>
      </header>

      <section className="rounded-xl bg-white p-5 shadow-sm ring-1 ring-slate-200">
        <h2 className="text-lg font-semibold">Dataset and evaluation</h2>
        <div className="mt-4 grid gap-4 md:grid-cols-[1fr_auto]">
          <label className="block text-sm font-medium text-slate-700">
            Dataset configuration path
            <input value={configPath} onChange={(event) => setConfigPath(event.target.value)} className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-slate-900" />
          </label>
          <button onClick={validateDataset} disabled={loading} className="self-end rounded-md bg-indigo-600 px-4 py-2 font-medium text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-slate-400">
            Validate dataset
          </button>
        </div>

        {validation?.valid && (
          <div className="mt-5 grid gap-4 border-t border-slate-200 pt-5 md:grid-cols-2">
            <label className="block text-sm font-medium text-slate-700">
              Student answer sheet selection
              <select value={studentId} onChange={(event) => setStudentId(event.target.value)} className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2">
                {validation.manifest.student_documents.map((student) => <option key={student.document_id} value={student.document_id}>{student.document_id}</option>)}
              </select>
            </label>
            <div className="text-sm text-slate-700">
              <p className="font-medium">Answer key selection</p>
              <p className="mt-2 rounded-md bg-slate-100 px-3 py-2">{selectedStudent?.answer_key_id || 'No answer key mapped'}</p>
              <p className="mt-1 text-xs text-slate-500">Dataset configuration maps the selected student answer sheet to its answer key.</p>
            </div>
          </div>
        )}
        <div className="mt-5 flex flex-wrap items-center gap-3">
          <button onClick={startEvaluation} disabled={loading || !validation?.valid || !studentId} className="rounded-md bg-emerald-600 px-4 py-2 font-medium text-white hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-slate-400">
            {loading ? 'Processing…' : 'Evaluate'}
          </button>
          {evaluation && <span className="text-sm text-slate-600">Status: <strong className="capitalize">{evaluation.status}</strong></span>}
        </div>
        {error && <p role="alert" className="mt-4 rounded-md bg-rose-50 px-3 py-2 text-sm text-rose-700">{error}</p>}
      </section>

      {results && (
        <section className="mt-8 space-y-6">
          <div className="rounded-xl bg-white p-5 shadow-sm ring-1 ring-slate-200">
            <h2 className="text-lg font-semibold">Evaluation results</h2>
            {questions.length === 0 ? (
              <div className="mt-4 rounded-md border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
                <strong>No questions detected.</strong> The current deterministic segmentation baseline needs visible question markers such as Q1, Q2, or 1. in the OCR text.
              </div>
            ) : (
              <>
                <p className="mt-2 text-2xl font-bold text-slate-950">{marks(results.grading.total_predicted_marks)} / {marks(results.grading.total_maximum_marks)}</p>
                <p className="text-sm text-slate-500">Method: {results.grading.grading_method} (transparent baseline, not semantic grading)</p>
                <div className="mt-5 space-y-4">
                  {questions.map((question) => <QuestionCard key={question.question_id} question={question} />)}
                </div>
              </>
            )}
          </div>
          <Benchmark metrics={benchmark} />
        </section>
      )}
    </main>
  )
}

function QuestionCard({ question }) {
  return (
    <article className="rounded-lg border border-slate-200 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <h3 className="font-semibold text-slate-950">{question.question_id}</h3>
        <p className="font-semibold text-indigo-700">{marks(question.predicted_marks)} / {marks(question.maximum_marks)}</p>
      </div>
      <p className="mt-1 text-sm text-slate-500">Similarity: {score(question.similarity_score)} · {question.grading_method}</p>
      <div className="mt-4 grid gap-3 md:grid-cols-2">
        <Answer label="Student answer" value={question.student_answer} />
        <Answer label="Reference answer" value={question.reference_answer} />
      </div>
      {question.figures?.length > 0 && <div className="mt-4 rounded-md bg-slate-50 p-3 text-sm text-slate-700"><strong>Associated figures</strong>{question.figures.map((figure, index) => <p key={index} className="mt-1">Page {figure.page_number}: {figure.element?.label || figure.element?.type || 'figure'}</p>)}</div>}
    </article>
  )
}

function Answer({ label, value }) {
  return <div><h4 className="text-sm font-medium text-slate-700">{label}</h4><p className="mt-1 whitespace-pre-wrap text-sm text-slate-600">{value || 'No answer available.'}</p></div>
}

function Benchmark({ metrics }) {
  if (!metrics || metrics.status !== 'ground_truth_available') {
    return <section className="rounded-xl bg-white p-5 shadow-sm ring-1 ring-slate-200"><h2 className="text-lg font-semibold">Benchmark</h2><p className="mt-2 text-sm text-slate-600">Ground truth unavailable.</p></section>
  }
  const values = [['MAE', metrics.mae], ['RMSE', metrics.rmse], ['Pearson', metrics.pearson_correlation], ['Spearman', metrics.spearman_correlation], ['Within 1 mark', metrics.agreement_within_1_mark]]
  return <section className="rounded-xl bg-white p-5 shadow-sm ring-1 ring-slate-200"><h2 className="text-lg font-semibold">Benchmark</h2><div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">{values.map(([label, value]) => <div key={label} className="rounded-md bg-slate-50 p-3"><p className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</p><p className="mt-1 text-lg font-semibold">{value === null ? 'N/A' : Number(value).toFixed(3)}</p></div>)}</div></section>
}
