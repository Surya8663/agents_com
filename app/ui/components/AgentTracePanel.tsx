import React from 'react';
import { motion } from 'framer-motion';
import { 
  Eye, 
  FileText, 
  Merge, 
  CheckCircle, 
  Clock, 
  AlertCircle,
  ChevronRight,
  Brain,
  Zap,
  Shield
} from 'lucide-react';

interface AgentTrace {
  agent: string;
  confidence: number;
  timestamp: string;
  insights: string[];
  duration?: number;
  status?: 'completed' | 'processing' | 'failed';
  input_size?: number;
  output_size?: number;
  warnings?: string[];
}

interface AgentTracePanelProps {
  traces: AgentTrace[];
  className?: string;
  onTraceClick?: (trace: AgentTrace) => void;
}

const AgentTracePanel: React.FC<AgentTracePanelProps> = ({
  traces,
  className = '',
  onTraceClick
}) => {
  const getAgentConfig = (agentName: string) => {
    const configs = {
      'VisionAgent': {
        icon: Eye,
        color: 'bg-purple-100 text-purple-600 border-purple-200',
        description: 'Analyzes document layout and spatial relationships',
        gradient: 'from-purple-500 to-pink-500'
      },
      'TextAgent': {
        icon: FileText,
        color: 'bg-green-100 text-green-600 border-green-200',
        description: 'Extracts semantic meaning from OCR text',
        gradient: 'from-green-500 to-emerald-500'
      },
      'FusionAgent': {
        icon: Merge,
        color: 'bg-blue-100 text-blue-600 border-blue-200',
        description: 'Combines visual and textual analysis',
        gradient: 'from-blue-500 to-cyan-500'
      },
      'ValidationAgent': {
        icon: CheckCircle,
        color: 'bg-amber-100 text-amber-600 border-amber-200',
        description: 'Validates accuracy and detects hallucinations',
        gradient: 'from-amber-500 to-orange-500'
      },
      'RAGAgent': {
        icon: Brain,
        color: 'bg-indigo-100 text-indigo-600 border-indigo-200',
        description: 'Retrieves relevant context from knowledge base',
        gradient: 'from-indigo-500 to-violet-500'
      },
      'ConfidenceEngine': {
        icon: Shield,
        color: 'bg-rose-100 text-rose-600 border-rose-200',
        description: 'Calculates confidence scores and risk assessment',
        gradient: 'from-rose-500 to-pink-500'
      }
    };

    return configs[agentName as keyof typeof configs] || {
      icon: Zap,
      color: 'bg-gray-100 text-gray-600 border-gray-200',
      description: 'Processing agent',
      gradient: 'from-gray-500 to-gray-600'
    };
  };

  const getStatusIcon = (status?: string) => {
    if (!status) return <Clock className="w-4 h-4 text-gray-500" />;
    
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'processing':
        return <Clock className="w-4 h-4 text-blue-500 animate-pulse" />;
      case 'failed':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      default:
        return <Clock className="w-4 h-4 text-gray-500" />;
    }
  };

  const formatDuration = (ms?: number) => {
    if (!ms) return 'N/A';
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit',
      second: '2-digit'
    });
  };

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Header */}
      <div className="p-4 border-b">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">Agent Trace</h2>
            <p className="text-sm text-gray-600">Real-time processing insights</p>
          </div>
          <div className="flex items-center space-x-2">
            <div className="text-sm text-gray-600">
              {traces.filter(t => t.status === 'completed').length}/{traces.length} completed
            </div>
          </div>
        </div>
      </div>

      {/* Agent traces */}
      <div className="flex-1 overflow-auto p-4 space-y-4">
        {traces.length === 0 ? (
          <div className="text-center py-8">
            <Zap className="w-12 h-12 text-gray-300 mx-auto mb-3" />
            <p className="text-gray-500">No agent traces available</p>
            <p className="text-sm text-gray-400 mt-1">Run document processing to see agent activity</p>
          </div>
        ) : (
          <div className="space-y-4">
            {traces.map((trace, index) => {
              const agentName = trace.agent || 'unknown';
              const config = getAgentConfig(agentName);
              const Icon = config.icon;
              const status = trace.status || 'processing';
              
              return (
                <motion.div
                  key={`${agentName}-${index}`}  // Fixed line 80
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className={`border rounded-xl p-4 cursor-pointer hover:shadow-md transition-all ${config.color} ${
                    onTraceClick ? 'hover:border-current' : ''
                  }`}
                  onClick={() => onTraceClick && onTraceClick(trace)}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <div className={`p-2 rounded-lg ${config.color.split(' ')[0]}`}>
                        <Icon className="w-5 h-5" />
                      </div>
                      <div>
                        <h3 className="font-semibold text-gray-900">{agentName}</h3>
                        <p className="text-sm text-gray-600">{config.description}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      {getStatusIcon(status)}
                      <span className="text-xs font-medium px-2 py-1 rounded-full bg-white/50">
                        {status}
                      </span>
                    </div>
                  </div>

                  {/* Confidence bar */}
                  <div className="mb-4">
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-700">Confidence</span>
                      <span className="font-semibold">
                        {Math.round(trace.confidence * 100)}%
                      </span>
                    </div>
                    <div className="h-2 bg-white/50 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full bg-linear-to-r ${config.gradient}`}
                        style={{ width: `${trace.confidence * 100}%` }}
                      />
                    </div>
                  </div>

                  {/* Insights */}
                  {trace.insights && trace.insights.length > 0 && (
                    <div className="mb-3">
                      <h4 className="text-sm font-medium text-gray-700 mb-2">Key Insights</h4>
                      <div className="space-y-1">
                        {trace.insights.slice(0, 3).map((insight, i) => (
                          <div key={i} className="flex items-start text-sm">
                            <ChevronRight className="w-4 h-4 mt-0.5 mr-2 shrink-0 text-gray-400" />
                            <span className="text-gray-600">{insight}</span>
                          </div>
                        ))}
                        {trace.insights.length > 3 && (
                          <div className="text-xs text-gray-500 mt-1">
                            +{trace.insights.length - 3} more insights
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Warnings */}
                  {trace.warnings && trace.warnings.length > 0 && (
                    <div className="mb-3">
                      <div className="flex items-center text-sm text-amber-700 mb-1">
                        <AlertCircle className="w-4 h-4 mr-1" />
                        <span className="font-medium">Warnings</span>
                      </div>
                      <div className="text-xs text-amber-600 bg-amber-50 p-2 rounded">
                        {trace.warnings[0]}
                        {trace.warnings.length > 1 && (
                          <span className="ml-1">
                            (+{trace.warnings.length - 1} more)
                          </span>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Metadata */}
                  <div className="flex items-center justify-between text-xs text-gray-500 pt-3 border-t">
                    <div className="flex items-center space-x-4">
                      <span title="Timestamp">
                        {formatTimestamp(trace.timestamp)}
                      </span>
                      {trace.duration && (
                        <span title="Processing duration">
                          {formatDuration(trace.duration)}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center space-x-2">
                      {trace.input_size && (
                        <span title="Input size">
                          In: {trace.input_size}
                        </span>
                      )}
                      {trace.output_size && (
                        <span title="Output size">
                          Out: {trace.output_size}
                        </span>
                      )}
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </div>
        )}
      </div>

      {/* Summary footer */}
      {traces.length > 0 && (
        <div className="p-4 border-t bg-linear-to-r from-gray-50 to-gray-100">
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900">
                {traces.length > 0 
                  ? Math.round(
                      traces.reduce((sum, t) => sum + t.confidence, 0) / traces.length * 100
                    )
                  : 0}%
              </div>
              <div className="text-xs text-gray-600">Avg Confidence</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900">
                {traces.filter(t => t.status === 'completed').length}
              </div>
              <div className="text-xs text-gray-600">Agents Completed</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900">
                {formatDuration(
                  traces.reduce((sum, t) => sum + (t.duration || 0), 0)
                )}
              </div>
              <div className="text-xs text-gray-600">Total Processing</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AgentTracePanel;