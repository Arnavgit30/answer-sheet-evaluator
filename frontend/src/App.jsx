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
  const [detailedQuestions, setDetailedQuestions] = useState([])
  const [ocrReviews, setOcrReviews] = useState([])
  const [gradingReviews, setGradingReviews] = useState([])
  const [activeTab, setActiveTab] = useState('evaluation') // 'evaluation' | 'ocr' | 'grading'
  const [activePageModal, setActivePageModal] = useState(null)
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
      setDetailedQuestions([])
      setOcrReviews([])
      setGradingReviews([])
      if (data.valid) setStudentId(data.manifest.student_documents[0]?.document_id || '')
      else setError(data.errors.join(' '))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const fetchDetails = async (evaluationId) => {
    try {
      const [res, qs, ocrRevs, grRevs] = await Promise.all([
        request(`/api/v1/evaluations/${evaluationId}/results`),
        request(`/api/v1/evaluations/${evaluationId}/questions`).catch(() => []),
        request(`/api/v1/evaluations/${evaluationId}/ocr-reviews`).catch(() => []),
        request(`/api/v1/evaluations/${evaluationId}/grading-reviews`).catch(() => []),
      ])
      setResults(res)
      setDetailedQuestions(qs)
      setOcrReviews(ocrRevs)
      setGradingReviews(grRevs)
    } catch (err) {
      setError(err.message)
    }
  }

  const pollEvaluation = async (evaluationId) => {
    const data = await request(`/api/v1/evaluations/${evaluationId}`)
    setEvaluation(data)
    if (data.status === 'completed') return fetchDetails(evaluationId)
    if (data.status === 'failed') throw new Error(data.error || 'Evaluation failed.')
    window.setTimeout(() => pollEvaluation(evaluationId).catch((err) => setError(err.message)), 1200)
  }

  const startEvaluation = async () => {
    if (!validation?.valid || !studentId) return setError('Validate a dataset and select a student answer sheet first.')
    setLoading(true)
    setError('')
    setResults(null)
    setDetailedQuestions([])
    try {
      const data = await request('/api/v1/evaluations', {
        method: 'POST',
        body: JSON.stringify({ config_path: configPath, student_document_id: studentId }),
      })
      setEvaluation(data)
      if (data.status === 'completed') await fetchDetails(data.evaluation_id)
      else if (data.status === 'failed') throw new Error(data.error || 'Evaluation failed.')
      else await pollEvaluation(data.evaluation_id)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleOcrAction = async (reviewId, action, correctedText) => {
    if (!evaluation?.evaluation_id) return
    try {
      const updated = await request(`/api/v1/evaluations/${evaluation.evaluation_id}/ocr-reviews/${reviewId}`, {
        method: 'POST',
        body: JSON.stringify({ action, corrected_text: correctedText }),
      })
      setOcrReviews((prev) => prev.map((r) => (r.review_id === reviewId ? updated : r)))
      // Refresh questions
      const qs = await request(`/api/v1/evaluations/${evaluation.evaluation_id}/questions`).catch(() => [])
      setDetailedQuestions(qs)
    } catch (err) {
      alert(`Error updating OCR review: ${err.message}`)
    }
  }

  const handleGradingAction = async (reviewId, action, humanMarks, notes) => {
    if (!evaluation?.evaluation_id) return
    try {
      const updated = await request(`/api/v1/evaluations/${evaluation.evaluation_id}/grading-reviews/${reviewId}`, {
        method: 'POST',
        body: JSON.stringify({ action, human_marks: humanMarks ? Number(humanMarks) : null, notes }),
      })
      setGradingReviews((prev) => prev.map((r) => (r.review_id === reviewId ? updated : r)))
      // Refresh questions & results
      const [qs, res] = await Promise.all([
        request(`/api/v1/evaluations/${evaluation.evaluation_id}/questions`).catch(() => []),
        request(`/api/v1/evaluations/${evaluation.evaluation_id}/results`).catch(() => null),
      ])
      if (qs) setDetailedQuestions(qs)
      if (res) setResults(res)
    } catch (err) {
      alert(`Error updating grading review: ${err.message}`)
    }
  }

  const pendingOcrCount = ocrReviews.filter((r) => r.reviewer_action === 'pending').length
  const pendingGradingCount = gradingReviews.filter((r) => r.reviewer_action === 'pending').length

  return (
    <main className="mx-auto min-h-screen max-w-6xl px-4 py-8 sm:px-6">
      <header className="mb-8">
        <p className="text-sm font-semibold uppercase tracking-[0.2em] text-indigo-600">GradeP v2 Evaluation Engine</p>
        <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-950">Answer Sheet Evaluator</h1>
        <p className="mt-2 max-w-3xl text-slate-600">
          Multimodal handwritten exam evaluation with llama.cpp local OCR, dual baseline and VLM grading, spatial provenance, and human-in-the-loop review.
        </p>
      </header>

      {/* Dataset config and trigger */}
      <section className="rounded-xl bg-white p-5 shadow-sm ring-1 ring-slate-200">
        <h2 className="text-lg font-semibold text-slate-900">Dataset and Evaluation Configuration</h2>
        <div className="mt-4 grid gap-4 md:grid-cols-[1fr_auto]">
          <label className="block text-sm font-medium text-slate-700">
            Dataset configuration path
            <input
              value={configPath}
              onChange={(e) => setConfigPath(e.target.value)}
              className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-slate-900 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
            />
          </label>
          <button
            onClick={validateDataset}
            disabled={loading}
            className="self-end rounded-md bg-indigo-600 px-4 py-2 font-medium text-white hover:bg-indigo-700 disabled:bg-slate-400"
          >
            Validate dataset
          </button>
        </div>

        {validation?.valid && (
          <div className="mt-5 grid gap-4 border-t border-slate-200 pt-5 md:grid-cols-2">
            <label className="block text-sm font-medium text-slate-700">
              Student answer sheet selection
              <select
                value={studentId}
                onChange={(e) => setStudentId(e.target.value)}
                className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-slate-900"
              >
                {validation.manifest.student_documents.map((student) => (
                  <option key={student.document_id} value={student.document_id}>
                    {student.document_id} ({student.page_or_file_count} pages)
                  </option>
                ))}
              </select>
            </label>
            <div className="text-sm text-slate-700">
              <p className="font-medium">Answer key mapping</p>
              <p className="mt-2 rounded-md bg-slate-100 px-3 py-2 font-mono text-xs">{selectedStudent?.answer_key_id || 'No answer key mapped'}</p>
              <p className="mt-1 text-xs text-slate-500">Automatically linked via dataset manifest.</p>
            </div>
          </div>
        )}

        <div className="mt-5 flex flex-wrap items-center gap-3">
          <button
            onClick={startEvaluation}
            disabled={loading || !validation?.valid || !studentId}
            className="rounded-md bg-emerald-600 px-5 py-2 font-medium text-white hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-slate-400 shadow-sm"
          >
            {loading ? 'Evaluating Sheet…' : 'Start Full Pipeline'}
          </button>
          {evaluation && (
            <span className="text-sm text-slate-600">
              Status: <strong className="capitalize text-slate-900">{evaluation.status}</strong>
            </span>
          )}
        </div>
        {error && <p role="alert" className="mt-4 rounded-md bg-rose-50 px-3 py-2 text-sm text-rose-700">{error}</p>}
      </section>

      {/* Navigation tabs for results & review gates */}
      {results && (
        <section className="mt-8 space-y-6">
          <div className="flex border-b border-slate-200">
            <button
              onClick={() => setActiveTab('evaluation')}
              className={`pb-3 px-4 font-medium text-sm border-b-2 transition-colors ${activeTab === 'evaluation'
                  ? 'border-indigo-600 text-indigo-600'
                  : 'border-transparent text-slate-500 hover:text-slate-700'
                }`}
            >
              Evaluated Questions ({detailedQuestions.length || results.grading?.questions?.length || 0})
            </button>
            <button
              onClick={() => setActiveTab('ocr')}
              className={`pb-3 px-4 font-medium text-sm border-b-2 flex items-center gap-2 transition-colors ${activeTab === 'ocr'
                  ? 'border-indigo-600 text-indigo-600'
                  : 'border-transparent text-slate-500 hover:text-slate-700'
                }`}
            >
              OCR Review Gate
              {pendingOcrCount > 0 && (
                <span className="rounded-full bg-amber-500 px-2 py-0.5 text-xs text-white font-bold">{pendingOcrCount}</span>
              )}
            </button>
            <button
              onClick={() => setActiveTab('grading')}
              className={`pb-3 px-4 font-medium text-sm border-b-2 flex items-center gap-2 transition-colors ${activeTab === 'grading'
                  ? 'border-indigo-600 text-indigo-600'
                  : 'border-transparent text-slate-500 hover:text-slate-700'
                }`}
            >
              Grading Review Gate
              {pendingGradingCount > 0 && (
                <span className="rounded-full bg-rose-500 px-2 py-0.5 text-xs text-white font-bold">{pendingGradingCount}</span>
              )}
            </button>
          </div>

          {/* TAB 1: Main Evaluation Detail */}
          {activeTab === 'evaluation' && (
            <div className="space-y-6">
              <div className="rounded-xl bg-white p-5 shadow-sm ring-1 ring-slate-200">
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <div>
                    <h2 className="text-lg font-semibold text-slate-900">Final Evaluated Marks</h2>
                    <p className="text-sm text-slate-500">
                      Total score evaluated with multimodal reasoning, semantic concepts, and deterministic fallback.
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-3xl font-extrabold text-indigo-600">
                      {marks(results.grading.total_predicted_marks)} / {marks(results.grading.total_maximum_marks)}
                    </p>
                    <p className="text-xs text-slate-500 font-mono">Method: {results.grading.grading_method}</p>
                  </div>
                </div>

                <div className="mt-6 space-y-4">
                  {detailedQuestions.length > 0
                    ? detailedQuestions.map((q) => (
                      <DetailedQuestionCard
                        key={q.question_id}
                        q={q}
                        onViewPage={(pageNum) => setActivePageModal({ evaluationId: evaluation.evaluation_id, pageNum })}
                      />
                    ))
                    : results.grading.questions.map((q) => <QuestionCard key={q.question_id} question={q} />)}
                </div>
              </div>

              {/* Benchmark comparison card */}
              <Benchmark metrics={results.benchmark} />
            </div>
          )}

          {/* TAB 2: OCR Review Gate */}
          {activeTab === 'ocr' && (
            <div className="rounded-xl bg-white p-5 shadow-sm ring-1 ring-slate-200">
              <h2 className="text-lg font-semibold text-slate-900">OCR Confidence & Human Review</h2>
              <p className="mt-1 text-sm text-slate-600">
                Low-confidence transcriptions flagged by character sanity and OCR heuristics. Verify or correct below.
              </p>
              {ocrReviews.length === 0 ? (
                <div className="mt-4 rounded-md border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-800">
                  All OCR extractions passed confidence thresholds! No manual corrections pending.
                </div>
              ) : (
                <div className="mt-4 space-y-4">
                  {ocrReviews.map((rev) => (
                    <OCRReviewCard
                      key={rev.review_id}
                      review={rev}
                      onAction={(action, text) => handleOcrAction(rev.review_id, action, text)}
                      onViewImage={() => setActivePageModal({ evaluationId: evaluation.evaluation_id, pageNum: 1 })}
                    />
                  ))}
                </div>
              )}
            </div>
          )}

          {/* TAB 3: Grading Review Gate */}
          {activeTab === 'grading' && (
            <div className="rounded-xl bg-white p-5 shadow-sm ring-1 ring-slate-200">
              <h2 className="text-lg font-semibold text-slate-900">Grading Confidence & Teacher Overrides</h2>
              <p className="mt-1 text-sm text-slate-600">
                Questions flagged for teacher review due to score divergence or ambiguous explanations.
              </p>
              {gradingReviews.length === 0 ? (
                <div className="mt-4 rounded-md border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-800">
                  All automated grades satisfied confidence criteria. No manual review items flagged.
                </div>
              ) : (
                <div className="mt-4 space-y-4">
                  {gradingReviews.map((rev) => (
                    <GradingReviewCard
                      key={rev.review_id}
                      review={rev}
                      onAction={(action, marks, notes) => handleGradingAction(rev.review_id, action, marks, notes)}
                    />
                  ))}
                </div>
              )}
            </div>
          )}
        </section>
      )}

      {/* Modal image viewer */}
      {activePageModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 p-4">
          <div className="relative max-h-[90vh] max-w-4xl overflow-auto rounded-lg bg-white p-4 shadow-2xl">
            <div className="flex items-center justify-between pb-3 border-b">
              <h3 className="font-semibold text-slate-900">Original Exam Page {activePageModal.pageNum}</h3>
              <button
                onClick={() => setActivePageModal(null)}
                className="rounded bg-slate-100 px-3 py-1 text-sm font-semibold text-slate-700 hover:bg-slate-200"
              >
                Close
              </button>
            </div>
            <img
              src={`${API_BASE_URL}/api/v1/evaluations/${activePageModal.evaluationId}/pages/${activePageModal.pageNum}`}
              alt={`Page ${activePageModal.pageNum}`}
              className="mt-3 max-h-[75vh] w-auto mx-auto object-contain rounded border"
              onError={(e) => {
                e.target.onerror = null
                e.target.src = ''
                alert('Page image preview not available.')
              }}
            />
          </div>
        </div>
      )}
    </main>
  )
}

function DetailedQuestionCard({ q, onViewPage }) {
  const ans = q.student_answer || {}
  const ref = q.reference_answer || {}
  const vlm = q.vlm_evaluation || {}
  const baseline = q.baseline_evaluation || {}

  const pages = ans.pages || []

  return (
    <article className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-slate-100 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-bold text-slate-950">{q.question_id}</h3>
            {pages.map((p) => (
              <button
                key={p}
                onClick={() => onViewPage(p)}
                className="rounded bg-indigo-50 px-2 py-0.5 text-xs font-semibold text-indigo-700 hover:bg-indigo-100"
              >
                View Page {p}
              </button>
            ))}
          </div>
          <p className="mt-1 text-xs text-slate-500">
            OCR Confidence: {ans.ocr_confidence !== undefined ? score(ans.ocr_confidence * 100) : '—'} · Baseline Match: {score(baseline.similarity_score)}
          </p>
        </div>
        <div className="text-right">
          <p className="text-xl font-bold text-indigo-700">
            {marks(vlm.marks_awarded)} / {marks(ref.maximum_marks ?? baseline.maximum_marks)}
          </p>
          <div className="mt-1 flex flex-wrap justify-end gap-1">
            <span className="inline-block rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-700">
              Method: {vlm.grading_method || 'llm'}
            </span>
            {(vlm.human_review_required || (vlm.status && vlm.status !== 'completed')) && (
              <span className="inline-block rounded-full bg-rose-100 px-2 py-0.5 text-[11px] font-semibold text-rose-800">
                Human Review Required
              </span>
            )}
          </div>
        </div>
      </div>

      {/* VLM Reasoning & Concepts */}
      {vlm.reasoning && (
        <div className="mt-3 rounded-md bg-slate-50 p-3 text-sm text-slate-800">
          <p className="font-semibold text-xs text-slate-600 uppercase tracking-wide">Evaluation Reasoning</p>
          <p className="mt-1">{vlm.reasoning}</p>

          <div className="mt-2 flex flex-wrap gap-2">
            {vlm.concepts_present?.map((c, i) => (
              <span key={i} className="rounded bg-emerald-100 px-2 py-0.5 text-xs text-emerald-800">
                ✓ {c}
              </span>
            ))}
            {vlm.concepts_missing?.map((c, i) => (
              <span key={i} className="rounded bg-rose-100 px-2 py-0.5 text-xs text-rose-800">
                ✗ {c}
              </span>
            ))}
            {vlm.errors?.map((err, i) => (
              <span key={i} className="rounded bg-amber-100 px-2 py-0.5 text-xs text-amber-800">
                ⚠ {err}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Side-by-side answers */}
      <div className="mt-4 grid gap-4 md:grid-cols-2">
        <div className="rounded-lg bg-slate-50/60 p-3 border border-slate-100">
          <h4 className="text-xs font-semibold uppercase text-slate-500 tracking-wider">Student Handwritten Answer</h4>
          <p className="mt-2 whitespace-pre-wrap font-mono text-xs text-slate-800">{ans.text || 'No transcribed text'}</p>
          {ans.formulas?.length > 0 && (
            <div className="mt-2 border-t pt-2 text-xs">
              <strong className="text-indigo-600">Extracted Formula:</strong>
              {ans.formulas.map((f, i) => <p key={i} className="font-mono text-slate-700">{f.raw_text}</p>)}
            </div>
          )}
        </div>
        <div className="rounded-lg bg-indigo-50/30 p-3 border border-indigo-100">
          <h4 className="text-xs font-semibold uppercase text-indigo-700 tracking-wider">Reference Key</h4>
          <p className="mt-2 whitespace-pre-wrap text-xs text-slate-800">{ref.answer_text || baseline.reference_answer || 'No reference'}</p>
        </div>
      </div>
    </article>
  )
}

function OCRReviewCard({ review, onAction, onViewImage }) {
  const [text, setText] = useState(review.original_text)

  return (
    <article className="rounded-lg border border-slate-200 p-4">
      <div className="flex items-center justify-between">
        <h4 className="font-semibold text-slate-900">{review.question_id} (Region: {review.region_id})</h4>
        <span className={`text-xs font-bold px-2 py-0.5 rounded capitalize ${review.reviewer_action === 'confirmed' ? 'bg-emerald-100 text-emerald-800' : review.reviewer_action === 'corrected' ? 'bg-indigo-100 text-indigo-800' : 'bg-amber-100 text-amber-800'}`}>
          {review.reviewer_action}
        </span>
      </div>
      <div className="mt-2">
        <label className="text-xs font-medium text-slate-600">Editable Transcription:</label>
        <textarea
          rows={3}
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="mt-1 w-full rounded border border-slate-300 p-2 text-xs font-mono"
        />
      </div>
      <div className="mt-3 flex gap-2">
        <button
          onClick={() => onAction('confirmed', text)}
          className="rounded bg-emerald-600 px-3 py-1 text-xs font-medium text-white hover:bg-emerald-700"
        >
          Confirm Text
        </button>
        <button
          onClick={() => onAction('corrected', text)}
          className="rounded bg-indigo-600 px-3 py-1 text-xs font-medium text-white hover:bg-indigo-700"
        >
          Save Correction
        </button>
      </div>
    </article>
  )
}

function GradingReviewCard({ review, onAction }) {
  const [marksVal, setMarksVal] = useState(review.human_marks ?? review.vlm_marks ?? '')
  const [notes, setNotes] = useState('')

  return (
    <article className="rounded-lg border border-slate-200 p-4">
      <div className="flex items-center justify-between">
        <h4 className="font-semibold text-slate-900">{review.question_id}</h4>
        <span className={`text-xs font-bold px-2 py-0.5 rounded capitalize ${review.reviewer_action === 'overridden' ? 'bg-indigo-100 text-indigo-800' : review.reviewer_action === 'confirmed' ? 'bg-emerald-100 text-emerald-800' : 'bg-rose-100 text-rose-800'}`}>
          {review.reviewer_action}
        </span>
      </div>
      <p className="mt-1 text-xs text-slate-600"><strong>VLM Grade:</strong> {marks(review.vlm_marks)} marks · <em>{review.vlm_reasoning}</em></p>
      <div className="mt-3 flex flex-wrap items-center gap-3">
        <label className="text-xs font-medium text-slate-700 flex items-center gap-2">
          Override Marks:
          <input
            type="number"
            step="0.5"
            value={marksVal}
            onChange={(e) => setMarksVal(e.target.value)}
            className="w-20 rounded border border-slate-300 px-2 py-1 text-xs"
          />
        </label>
        <input
          placeholder="Override notes/reasons..."
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          className="flex-1 rounded border border-slate-300 px-2 py-1 text-xs"
        />
        <button
          onClick={() => onAction('overridden', marksVal, notes)}
          className="rounded bg-indigo-600 px-3 py-1 text-xs font-medium text-white hover:bg-indigo-700"
        >
          Apply Override
        </button>
        <button
          onClick={() => onAction('confirmed', review.vlm_marks, 'Confirmed VLM')}
          className="rounded bg-slate-200 px-3 py-1 text-xs font-medium text-slate-700 hover:bg-slate-300"
        >
          Confirm VLM Grade
        </button>
      </div>
    </article>
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
    </article>
  )
}

function Answer({ label, value }) {
  return (
    <div>
      <h4 className="text-sm font-medium text-slate-700">{label}</h4>
      <p className="mt-1 whitespace-pre-wrap text-sm text-slate-600">{value || 'No answer available.'}</p>
    </div>
  )
}

function Benchmark({ metrics }) {
  if (!metrics || metrics.status !== 'ground_truth_available') {
    return (
      <section className="rounded-xl bg-white p-5 shadow-sm ring-1 ring-slate-200">
        <h2 className="text-lg font-semibold text-slate-900">Benchmark Metrics</h2>
        <p className="mt-2 text-sm text-slate-600">Ground truth human marks unavailable for this dataset.</p>
      </section>
    )
  }
  const values = [
    ['MAE', metrics.mae],
    ['RMSE', metrics.rmse],
    ['Pearson', metrics.pearson_correlation],
    ['Spearman', metrics.spearman_correlation],
    ['Within 1 mark', metrics.agreement_within_1_mark],
  ]
  return (
    <section className="rounded-xl bg-white p-5 shadow-sm ring-1 ring-slate-200">
      <h2 className="text-lg font-semibold text-slate-900">Benchmark vs Human Evaluator</h2>
      <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        {values.map(([label, value]) => (
          <div key={label} className="rounded-md bg-slate-50 p-3">
            <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</p>
            <p className="mt-1 text-lg font-semibold">{value === null ? 'N/A' : Number(value).toFixed(3)}</p>
          </div>
        ))}
      </div>
    </section>
  )
}
