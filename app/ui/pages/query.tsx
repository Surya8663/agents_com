import React, { useState } from 'react';
import { useRouter } from 'next/router';
import { motion } from 'framer-motion';
import { Search, Filter, Download, Eye, MessageSquare, BarChart3 } from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';

interface QueryResult {
  query: string;
  answer: string;
  confidence: number;
  sources: Array<{
    page: number;
    confidence: number;
    type: string;
    content: string;
  }>;
  needs_review: boolean;
}

const QueryPage: React.FC = () => {
  const router = useRouter();
  const [query, setQuery] = useState('');
  const [documentId, setDocumentId] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<QueryResult[]>([]);
  const [selectedResult, setSelectedResult] = useState<QueryResult | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!query.trim()) {
      toast.error('Please enter a query');
      return;
    }

    setIsLoading(true);
    
    try {
      const response = await axios.post('/api/rag/query/simple', {
        query,
        document_id: documentId || undefined,
        top_k: 5
      });

      const newResult = response.data;
      setResults(prev => [newResult, ...prev]);
      setSelectedResult(newResult);
      
      toast.success('Query processed successfully');
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Query failed');
      console.error('Query error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setQuery('');
    setDocumentId('');
    setSelectedResult(null);
  };

  const handleExport = async () => {
    if (!selectedResult) return;

    const data = {
      timestamp: new Date().toISOString(),
      ...selectedResult
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `query-result-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    toast.success('Result exported');
  };

  return (
    <div className="min-h-screen bg-linear-to-br from-gray-50 to-blue-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Multi-Modal Document Query
          </h1>
          <p className="text-gray-600">
            Ask questions about your documents using natural language. The system will search across text, tables, figures, and signatures.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Query panel */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-2"
          >
            <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">
              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Your Question
                  </label>
                  <div className="relative">
                    <Search className="absolute left-3 top-3.5 w-5 h-5 text-gray-400" />
                    <textarea
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="Example: What is the total amount on the invoice? Find all tables in the document. Extract signature locations..."
                      className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none h-32"
                      disabled={isLoading}
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Document ID (Optional)
                    </label>
                    <input
                      type="text"
                      value={documentId}
                      onChange={(e) => setDocumentId(e.target.value)}
                      placeholder="Leave empty to search all documents"
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      disabled={isLoading}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Query Type
                    </label>
                    <div className="flex space-x-2">
                      <button
                        type="button"
                        className="flex-1 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
                      >
                        <Filter className="w-4 h-4 inline mr-2" />
                        Filter
                      </button>
                      <button
                        type="button"
                        className="flex-1 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
                      >
                        <Eye className="w-4 h-4 inline mr-2" />
                        Visual
                      </button>
                    </div>
                  </div>
                </div>

                <div className="flex justify-end space-x-3 pt-4">
                  <button
                    type="button"
                    onClick={handleClear}
                    className="px-6 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                    disabled={isLoading}
                  >
                    Clear
                  </button>
                  <button
                    type="submit"
                    disabled={isLoading || !query.trim()}
                    className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    {isLoading ? (
                      <span className="flex items-center">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                        Processing...
                      </span>
                    ) : (
                      <span className="flex items-center">
                        <Search className="w-4 h-4 mr-2" />
                        Ask Question
                      </span>
                    )}
                  </button>
                </div>
              </form>
            </div>

            {/* Query history */}
            {results.length > 0 && (
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <h2 className="text-xl font-semibold text-gray-900 mb-4">
                  Recent Queries
                </h2>
                <div className="space-y-3">
                  {results.map((result, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className={`p-4 rounded-lg border cursor-pointer transition-all ${
                        selectedResult?.query === result.query
                          ? 'border-blue-500 bg-blue-50'
                          : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                      }`}
                      onClick={() => setSelectedResult(result)}
                    >
                      <div className="flex justify-between items-start">
                        <div className="flex-1">
                          <div className="flex items-center space-x-2 mb-2">
                            <MessageSquare className="w-4 h-4 text-gray-400" />
                            <h3 className="font-medium text-gray-900">
                              {result.query.length > 60
                                ? `${result.query.substring(0, 60)}...`
                                : result.query}
                            </h3>
                          </div>
                          <div className="flex items-center space-x-4 text-sm">
                            <span className="text-gray-600">
                              {result.sources.length} sources
                            </span>
                            <span className={`px-2 py-0.5 rounded-full text-xs ${
                              result.confidence > 0.8
                                ? 'bg-green-100 text-green-800'
                                : result.confidence > 0.6
                                ? 'bg-yellow-100 text-yellow-800'
                                : 'bg-red-100 text-red-800'
                            }`}>
                              {Math.round(result.confidence * 100)}% confidence
                            </span>
                            {result.needs_review && (
                              <span className="px-2 py-0.5 rounded-full text-xs bg-amber-100 text-amber-800">
                                Needs review
                              </span>
                            )}
                          </div>
                        </div>
                        <div className="text-xs text-gray-500">
                          {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>

          {/* Results panel */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-1"
          >
            <div className="sticky top-6">
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <div className="flex justify-between items-center mb-6">
                  <h2 className="text-xl font-semibold text-gray-900">
                    Query Results
                  </h2>
                  {selectedResult && (
                    <button
                      onClick={handleExport}
                      className="p-2 hover:bg-gray-100 rounded-lg"
                      title="Export result"
                    >
                      <Download className="w-5 h-5" />
                    </button>
                  )}
                </div>

                {selectedResult ? (
                  <div className="space-y-6">
                    {/* Confidence meter */}
                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm font-medium text-gray-700">
                          Confidence Score
                        </span>
                        <span className="text-lg font-bold text-gray-900">
                          {Math.round(selectedResult.confidence * 100)}%
                        </span>
                      </div>
                      <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-linear-to-r from-red-500 via-yellow-500 to-green-500"
                          style={{ width: `${selectedResult.confidence * 100}%` }}
                        />
                      </div>
                    </div>

                    {/* Answer */}
                    <div>
                      <h3 className="text-sm font-medium text-gray-700 mb-2">
                        Answer
                      </h3>
                      <div className="p-4 bg-blue-50 rounded-lg border border-blue-100">
                        <p className="text-gray-800">{selectedResult.answer}</p>
                      </div>
                    </div>

                    {/* Sources */}
                    <div>
                      <h3 className="text-sm font-medium text-gray-700 mb-2">
                        Sources ({selectedResult.sources.length})
                      </h3>
                      <div className="space-y-3 max-h-80 overflow-auto">
                        {selectedResult.sources.map((source, index) => (
                          <div
                            key={index}
                            className="p-3 border border-gray-200 rounded-lg"
                          >
                            <div className="flex justify-between items-start mb-2">
                              <div className="flex items-center space-x-2">
                                <BarChart3 className="w-4 h-4 text-gray-400" />
                                <span className="text-sm font-medium">
                                  Page {source.page}
                                </span>
                              </div>
                              <span className={`px-2 py-0.5 rounded text-xs ${
                                source.confidence > 0.8
                                  ? 'bg-green-100 text-green-800'
                                  : 'bg-yellow-100 text-yellow-800'
                              }`}>
                                {Math.round(source.confidence * 100)}%
                              </span>
                            </div>
                            <p className="text-sm text-gray-600">
                              {source.content.length > 100
                                ? `${source.content.substring(0, 100)}...`
                                : source.content}
                            </p>
                            <div className="mt-2 text-xs text-gray-500">
                              Type: {source.type.replace('_', ' ')}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Actions */}
                    {selectedResult.needs_review && (
                      <div className="p-4 bg-amber-50 border border-amber-200 rounded-lg">
                        <h4 className="font-medium text-amber-800 mb-2">
                          ⚠️ Human Review Recommended
                        </h4>
                        <p className="text-sm text-amber-700 mb-3">
                          Confidence is below threshold. Consider reviewing this extraction.
                        </p>
                        <button
                          onClick={() => router.push('/review')}
                          className="w-full py-2 bg-amber-100 text-amber-800 rounded-lg hover:bg-amber-200 transition-colors"
                        >
                          Send for Review
                        </button>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <Search className="w-12 h-12 text-gray-300 mx-auto mb-4" />
                    <p className="text-gray-500">
                      Submit a query to see results here
                    </p>
                  </div>
                )}
              </div>

              {/* Tips */}
              <div className="mt-6 bg-linear-to-r from-blue-50 to-indigo-50 rounded-2xl p-6">
                <h3 className="font-semibold text-gray-900 mb-3">
                  💡 Query Tips
                </h3>
                <ul className="space-y-2 text-sm text-gray-700">
                  <li>• Be specific about what you're looking for</li>
                  <li>• Mention document sections if known</li>
                  <li>• Ask about tables, figures, or signatures</li>
                  <li>• Include document ID for targeted search</li>
                  <li>• Use natural language - the system understands context</li>
                </ul>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default QueryPage;