import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { motion, AnimatePresence } from 'framer-motion';
import {
  CheckCircle,
  XCircle,
  Edit,
  Eye,
  Download,
  Filter,
  Clock,
  AlertCircle,
  RefreshCw,
  Send,
  FileText,
  BarChart3,
  Users,
  TrendingUp
} from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';
import DocumentViewer from '../components/DocumentViewer';
import ConfidenceMeter from '../components/ConfidenceMeter';

interface ReviewItem {
  id: string;
  document_id: string;
  field_name: string;
  extracted_value: any;
  confidence: number;
  reason_for_review: string;
  page_number?: number;
  bbox?: { x1: number; y1: number; x2: number; y2: number };
  agent_sources: string[];
  timestamp: string;
  status: 'pending' | 'approved' | 'rejected' | 'edited';
  reviewer_notes?: string;
  corrected_value?: any;
  document_title?: string;
}

interface ReviewStats {
  total_reviews: number;
  by_status: Record<string, number>;
  by_document: Record<string, number>;
  avg_confidence: number;
  pending_count: number;
  completion_rate: number;
}

const ReviewPage: React.FC = () => {
  const router = useRouter();
  const [reviewItems, setReviewItems] = useState<ReviewItem[]>([]);
  const [selectedItem, setSelectedItem] = useState<ReviewItem | null>(null);
  const [stats, setStats] = useState<ReviewStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isProcessing, setIsProcessing] = useState(false);
  const [filters, setFilters] = useState({
    status: 'pending' as string,
    confidenceMin: 0,
    confidenceMax: 100,
    documentId: ''
  });
  const [correctionValue, setCorrectionValue] = useState('');

  // Load review queue
  const loadReviewQueue = async () => {
    setIsLoading(true);
    try {
      const response = await axios.get(`/api/review/queue?status=${filters.status}&limit=100`);
      setReviewItems(response.data);
      
      // Load stats
      const statsResponse = await axios.get('/api/review/stats');
      setStats(statsResponse.data);
      
      toast.success('Review queue loaded');
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to load review queue');
      console.error('Load error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Load on mount and when filters change
  useEffect(() => {
    loadReviewQueue();
  }, [filters.status]);

  // Handle review action
  const handleReviewAction = async (action: 'approve' | 'reject' | 'edit', notes?: string) => {
    if (!selectedItem) return;

    setIsProcessing(true);
    try {
      await axios.post('/api/review/process', {
        review_id: selectedItem.id,
        action,
        corrected_value: action === 'edit' ? correctionValue : undefined,
        reviewer_notes: notes
      });

      toast.success(`Item ${action}d successfully`);
      
      // Refresh queue
      await loadReviewQueue();
      
      // Clear selection if action completed
      setSelectedItem(null);
      setCorrectionValue('');
      
    } catch (error: any) {
      toast.error(error.response?.data?.detail || `Failed to ${action} item`);
      console.error('Action error:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  // Format timestamp
  const formatTimeAgo = (timestamp: string) => {
    const now = new Date();
    const time = new Date(timestamp);
    const diffMs = now.getTime() - time.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${diffDays}d ago`;
  };

  // Get status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending': return 'bg-yellow-100 text-yellow-800';
      case 'approved': return 'bg-green-100 text-green-800';
      case 'rejected': return 'bg-red-100 text-red-800';
      case 'edited': return 'bg-blue-100 text-blue-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  // Get status icon
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending': return Clock;
      case 'approved': return CheckCircle;
      case 'rejected': return XCircle;
      case 'edited': return Edit;
      default: return Clock;
    }
  };

  // Filter items
  const filteredItems = reviewItems.filter(item => {
    const confidencePercent = Math.round(item.confidence * 100);
    return (
      confidencePercent >= filters.confidenceMin &&
      confidencePercent <= filters.confidenceMax &&
      (filters.documentId === '' || item.document_id.includes(filters.documentId))
    );
  });

  return (
    <div className="min-h-screen bg-linear-to-br from-gray-50 to-amber-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-gray-900 mb-2">
                Human Review Dashboard
              </h1>
              <p className="text-gray-600">
                Review and validate document extractions with low confidence scores
              </p>
            </div>
            <button
              onClick={loadReviewQueue}
              disabled={isLoading}
              className="px-4 py-2 bg-white border rounded-lg hover:bg-gray-50 flex items-center space-x-2"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
              <span>Refresh</span>
            </button>
          </div>
        </motion.div>

        {/* Stats overview */}
        {stats && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8"
          >
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-2xl font-bold text-gray-900">
                      {stats.pending_count}
                    </div>
                    <div className="text-sm text-gray-600">Pending Reviews</div>
                  </div>
                  <div className="p-3 bg-amber-100 rounded-lg">
                    <Clock className="w-6 h-6 text-amber-600" />
                  </div>
                </div>
                <div className="mt-4">
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-amber-500 rounded-full"
                      style={{ width: `${(stats.pending_count / stats.total_reviews) * 100}%` }}
                    />
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-2xl shadow-lg p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-2xl font-bold text-gray-900">
                      {Math.round(stats.avg_confidence * 100)}%
                    </div>
                    <div className="text-sm text-gray-600">Avg Confidence</div>
                  </div>
                  <div className="p-3 bg-blue-100 rounded-lg">
                    <BarChart3 className="w-6 h-6 text-blue-600" />
                  </div>
                </div>
                <div className="mt-4">
                  <ConfidenceMeter 
                    confidence={stats.avg_confidence}
                    size="sm"
                    showLabel={false}
                  />
                </div>
              </div>

              <div className="bg-white rounded-2xl shadow-lg p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-2xl font-bold text-gray-900">
                      {stats.total_reviews}
                    </div>
                    <div className="text-sm text-gray-600">Total Items</div>
                  </div>
                  <div className="p-3 bg-green-100 rounded-lg">
                    <FileText className="w-6 h-6 text-green-600" />
                  </div>
                </div>
                <div className="mt-4">
                  <div className="flex items-center text-sm text-green-600">
                    <TrendingUp className="w-4 h-4 mr-1" />
                    <span>{stats.completion_rate.toFixed(1)}% completion rate</span>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-2xl shadow-lg p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-2xl font-bold text-gray-900">
                      {Object.keys(stats.by_document).length}
                    </div>
                    <div className="text-sm text-gray-600">Documents</div>
                  </div>
                  <div className="p-3 bg-purple-100 rounded-lg">
                    <Users className="w-6 h-6 text-purple-600" />
                  </div>
                </div>
                <div className="mt-4 text-sm text-gray-600">
                  Across {Object.keys(stats.by_status).length} status categories
                </div>
              </div>
            </div>
          </motion.div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left panel - Review queue */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-2"
          >
            <div className="bg-white rounded-2xl shadow-lg overflow-hidden">
              {/* Filters */}
              <div className="p-6 border-b">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-semibold text-gray-900">
                    Review Queue
                  </h2>
                  <div className="flex items-center space-x-2">
                    <Filter className="w-5 h-5 text-gray-400" />
                    <span className="text-sm text-gray-600">
                      {filteredItems.length} items
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Status
                    </label>
                    <select
                      value={filters.status}
                      onChange={(e) => setFilters({...filters, status: e.target.value})}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="pending">Pending</option>
                      <option value="approved">Approved</option>
                      <option value="rejected">Rejected</option>
                      <option value="edited">Edited</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Confidence Range
                    </label>
                    <div className="flex items-center space-x-2">
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={filters.confidenceMin}
                        onChange={(e) => setFilters({...filters, confidenceMin: parseInt(e.target.value)})}
                        className="flex-1"
                      />
                      <span className="text-sm">
                        {filters.confidenceMin}% - {filters.confidenceMax}%
                      </span>
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Document ID
                    </label>
                    <input
                      type="text"
                      value={filters.documentId}
                      onChange={(e) => setFilters({...filters, documentId: e.target.value})}
                      placeholder="Filter by document..."
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                </div>
              </div>

              {/* Review items list */}
              <div className="divide-y max-h-[600px] overflow-auto">
                {isLoading ? (
                  <div className="p-12 text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
                    <p className="mt-4 text-gray-600">Loading review queue...</p>
                  </div>
                ) : filteredItems.length === 0 ? (
                  <div className="p-12 text-center">
                    <CheckCircle className="w-16 h-16 text-green-400 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">
                      No items to review
                    </h3>
                    <p className="text-gray-600">
                      {filters.status === 'pending' 
                        ? 'All pending items have been reviewed!'
                        : 'No items match your filters.'}
                    </p>
                  </div>
                ) : (
                  <AnimatePresence>
                    {filteredItems.map((item, index) => {
                      const StatusIcon = getStatusIcon(item.status);
                      
                      return (
                        <motion.div
                          key={item.id}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -20 }}
                          transition={{ delay: index * 0.05 }}
                          className={`p-6 cursor-pointer hover:bg-gray-50 transition-colors ${
                            selectedItem?.id === item.id ? 'bg-blue-50 border-l-4 border-blue-500' : ''
                          }`}
                          onClick={() => setSelectedItem(item)}
                        >
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <div className="flex items-center space-x-3 mb-3">
                                <div className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(item.status)}`}>
                                  <div className="flex items-center space-x-1">
                                    <StatusIcon className="w-3 h-3" />
                                    <span>{item.status.toUpperCase()}</span>
                                  </div>
                                </div>
                                <span className="text-sm text-gray-600">
                                  {formatTimeAgo(item.timestamp)}
                                </span>
                              </div>

                              <h3 className="font-semibold text-gray-900 mb-2">
                                {item.field_name}
                              </h3>
                              
                              <div className="mb-3">
                                <div className="text-sm text-gray-700 mb-1">
                                  Extracted Value:
                                </div>
                                <div className="p-3 bg-gray-100 rounded-lg">
                                  <code className="text-sm font-mono">
                                    {typeof item.extracted_value === 'object'
                                      ? JSON.stringify(item.extracted_value, null, 2)
                                      : String(item.extracted_value)}
                                  </code>
                                </div>
                              </div>

                              <div className="flex items-center justify-between">
                                <div className="flex items-center space-x-4">
                                  <div className="text-sm">
                                    <span className="text-gray-600">Document: </span>
                                    <span className="font-medium">{item.document_id.substring(0, 8)}...</span>
                                  </div>
                                  {item.page_number && (
                                    <div className="text-sm">
                                      <span className="text-gray-600">Page: </span>
                                      <span className="font-medium">{item.page_number}</span>
                                    </div>
                                  )}
                                </div>

                                <div className="flex items-center space-x-3">
                                  <ConfidenceMeter
                                    confidence={item.confidence}
                                    size="sm"
                                    showLabel={false}
                                  />
                                  <span className="text-sm font-medium">
                                    {Math.round(item.confidence * 100)}%
                                  </span>
                                </div>
                              </div>
                            </div>
                          </div>
                        </motion.div>
                      );
                    })}
                  </AnimatePresence>
                )}
              </div>
            </div>
          </motion.div>

          {/* Right panel - Review interface */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-1"
          >
            <div className="sticky top-6 space-y-6">
              {/* Review actions panel */}
              {selectedItem ? (
                <div className="bg-white rounded-2xl shadow-lg p-6">
                  <div className="mb-6">
                    <h2 className="text-xl font-semibold text-gray-900 mb-2">
                      Review Extraction
                    </h2>
                    <p className="text-sm text-gray-600">
                      Validate or correct the extracted value
                    </p>
                  </div>

                  {/* Field info */}
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Field Name
                      </label>
                      <div className="p-3 bg-gray-50 rounded-lg">
                        <span className="font-medium">{selectedItem.field_name}</span>
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Extracted Value
                      </label>
                      <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg">
                        <code className="text-sm font-mono break-all">
                          {typeof selectedItem.extracted_value === 'object'
                            ? JSON.stringify(selectedItem.extracted_value, null, 2)
                            : String(selectedItem.extracted_value)}
                        </code>
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Reason for Review
                      </label>
                      <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                        <div className="flex items-start">
                          <AlertCircle className="w-5 h-5 text-red-500 mr-2 shrink-0" />
                          <span className="text-sm text-red-700">
                            {selectedItem.reason_for_review}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Confidence Score
                      </label>
                      <ConfidenceMeter
                        confidence={selectedItem.confidence}
                        size="lg"
                        showDetails={true}
                      />
                    </div>

                    {/* Correction input */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Corrected Value (Optional)
                      </label>
                      <textarea
                        value={correctionValue}
                        onChange={(e) => setCorrectionValue(e.target.value)}
                        placeholder="Enter corrected value if needed..."
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent h-32"
                      />
                    </div>

                    {/* Review notes */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Review Notes
                      </label>
                      <textarea
                        placeholder="Add notes about your decision..."
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent h-24"
                      />
                    </div>

                    {/* Action buttons */}
                    <div className="grid grid-cols-3 gap-3 pt-4">
                      <button
                        onClick={() => handleReviewAction('approve')}
                        disabled={isProcessing}
                        className="py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 transition-colors"
                      >
                        <div className="flex items-center justify-center">
                          <CheckCircle className="w-5 h-5 mr-2" />
                          Approve
                        </div>
                      </button>

                      <button
                        onClick={() => handleReviewAction('reject')}
                        disabled={isProcessing}
                        className="py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 transition-colors"
                      >
                        <div className="flex items-center justify-center">
                          <XCircle className="w-5 h-5 mr-2" />
                          Reject
                        </div>
                      </button>

                      <button
                        onClick={() => handleReviewAction('edit', correctionValue)}
                        disabled={isProcessing || !correctionValue.trim()}
                        className="py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
                      >
                        <div className="flex items-center justify-center">
                          <Edit className="w-5 h-5 mr-2" />
                          Edit
                        </div>
                      </button>
                    </div>

                    {/* Additional actions */}
                    <div className="flex space-x-3">
                      <button
                        onClick={() => router.push(`/query?document=${selectedItem.document_id}`)}
                        className="flex-1 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                      >
                        <div className="flex items-center justify-center">
                          <Eye className="w-4 h-4 mr-2" />
                          View Document
                        </div>
                      </button>
                      <button
                        onClick={() => {
                          // Navigate to document view with region highlighted
                          router.push(`/view/${selectedItem.document_id}?page=${selectedItem.page_number}&highlight=${selectedItem.field_name}`);
                        }}
                        className="flex-1 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                      >
                        <div className="flex items-center justify-center">
                          <Send className="w-4 h-4 mr-2" />
                          Jump to Page
                        </div>
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="bg-white rounded-2xl shadow-lg p-12 text-center">
                  <Eye className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">
                    Select an Item to Review
                  </h3>
                  <p className="text-gray-600">
                    Choose an item from the queue to view details and take action
                  </p>
                </div>
              )}

              {/* Quick stats */}
              <div className="bg-linear-to-r from-blue-50 to-indigo-50 rounded-2xl p-6">
                <h3 className="font-semibold text-gray-900 mb-4">
                  📈 Review Guidelines
                </h3>
                <ul className="space-y-3 text-sm text-gray-700">
                  <li className="flex items-start">
                    <CheckCircle className="w-4 h-4 text-green-500 mr-2 mt-0.5 shrink-0" />
                    <span><strong>Approve</strong> if extraction is correct and complete</span>
                  </li>
                  <li className="flex items-start">
                    <XCircle className="w-4 h-4 text-red-500 mr-2 mt-0.5 shrink-0" />
                    <span><strong>Reject</strong> if extraction is completely wrong</span>
                  </li>
                  <li className="flex items-start">
                    <Edit className="w-4 h-4 text-blue-500 mr-2 mt-0.5 shrink-0" />
                    <span><strong>Edit</strong> if extraction is partially correct</span>
                  </li>
                  <li className="flex items-start">
                    <AlertCircle className="w-4 h-4 text-amber-500 mr-2 mt-0.5 shrink-0" />
                    <span>Always add notes when rejecting or editing</span>
                  </li>
                  <li className="flex items-start">
                    <TrendingUp className="w-4 h-4 text-purple-500 mr-2 mt-0.5 shrink-0" />
                    <span>Your feedback improves the system's accuracy</span>
                  </li>
                </ul>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default ReviewPage;