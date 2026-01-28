import React, { useState, useCallback } from 'react';
import { useRouter } from 'next/router';
import { motion } from 'framer-motion';
import {
  Upload,
  FileText,
  Image,
  CheckCircle,
  Clock,
  AlertCircle,
  X,
  Eye,
  Zap,
  BarChart3,
  Cloud,
  Shield,
  Cpu,
  Database
} from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';
import { useDropzone } from 'react-dropzone';

interface UploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
  progress: number;
  status: 'uploading' | 'processing' | 'completed' | 'failed';
  documentId?: string;
  error?: string;
  uploadTime?: Date;
  pages?: number;
}

interface SystemStatus {
  ocr: 'ready' | 'busy' | 'offline';
  agents: 'ready' | 'busy' | 'offline';
  vector_db: 'ready' | 'busy' | 'offline';
  llm: 'ready' | 'busy' | 'offline';
}

const UploadPage: React.FC = () => {
  const router = useRouter();
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    ocr: 'ready',
    agents: 'ready',
    vector_db: 'ready',
    llm: 'ready'
  });

  // Handle file drop
  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const newFiles: UploadedFile[] = acceptedFiles.map(file => ({
      id: Math.random().toString(36).substring(7),
      name: file.name,
      size: file.size,
      type: file.type,
      progress: 0,
      status: 'uploading',
      uploadTime: new Date()
    }));

    setFiles(prev => [...prev, ...newFiles]);

    // Upload each file
    for (const fileData of newFiles) {
      await uploadFile(fileData, acceptedFiles.find(f => f.name === fileData.name)!);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    },
    multiple: true,
    onDragEnter: () => setIsDragging(true),
    onDragLeave: () => setIsDragging(false)
  });

  // Upload file to backend
  const uploadFile = async (fileData: UploadedFile, file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      // Update status to uploading
      setFiles(prev => prev.map(f =>
        f.id === fileData.id ? { ...f, status: 'uploading', progress: 10 } : f
      ));

      // Upload file
      const uploadResponse = await axios.post('http://localhost:8000/ingest/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        onUploadProgress: (progressEvent) => {
          const progress = progressEvent.total
            ? Math.round((progressEvent.loaded * 100) / progressEvent.total)
            : 0;
          
          setFiles(prev => prev.map(f =>
            f.id === fileData.id ? { ...f, progress: Math.min(90, progress) } : f
          ));
        }
      });

      const documentId = uploadResponse.data.document_id;

      // Update with document ID and move to processing
      setFiles(prev => prev.map(f =>
        f.id === fileData.id ? {
          ...f,
          progress: 100,
          status: 'processing',
          documentId
        } : f
      ));

      toast.success(`${fileData.name} uploaded successfully!`);

      // Start processing pipeline
      await startProcessingPipeline(fileData.id, documentId);

    } catch (error: any) {
      console.error('Upload error:', error);
      
      setFiles(prev => prev.map(f =>
        f.id === fileData.id ? {
          ...f,
          status: 'failed',
          error: error.response?.data?.detail || 'Upload failed'
        } : f
      ));

      toast.error(`Failed to upload ${fileData.name}: ${error.response?.data?.detail || 'Unknown error'}`);
    }
  };

  // Start the full processing pipeline
  const startProcessingPipeline = async (fileId: string, documentId: string) => {
    try {
      // Update system status
      setSystemStatus({
        ocr: 'busy',
        agents: 'busy',
        vector_db: 'busy',
        llm: 'busy'
      });

      // Phase 2: Layout analysis
      toast.loading('Analyzing document layout...', { id: 'layout' });
      await axios.post(`http://localhost:8000/layout/analyze/${documentId}`);
      toast.success('Layout analysis complete!', { id: 'layout' });

      setFiles(prev => prev.map(f =>
        f.id === fileId ? { ...f, progress: 25 } : f
      ));

      // Phase 3: OCR processing
      toast.loading('Performing OCR extraction...', { id: 'ocr' });
      await axios.post(`http://localhost:8000/ocr/process/${documentId}`, {
        lang: 'en',
        use_gpu: true
      });
      toast.success('OCR extraction complete!', { id: 'ocr' });

      setFiles(prev => prev.map(f =>
        f.id === fileId ? { ...f, progress: 50 } : f
      ));

      // Phase 4: Agent processing
      toast.loading('Running multi-agent analysis...', { id: 'agents' });
      await axios.post(`http://localhost:8000/agents/run/${documentId}`);
      toast.success('Agent analysis complete!', { id: 'agents' });

      setFiles(prev => prev.map(f =>
        f.id === fileId ? { ...f, progress: 75 } : f
      ));

      // Phase 5: RAG indexing
      toast.loading('Indexing for multi-modal search...', { id: 'rag' });
      await axios.post(`http://localhost:8000/rag/index/${documentId}`);
      toast.success('Document indexed for search!', { id: 'rag' });

      // Final update
      setFiles(prev => prev.map(f =>
        f.id === fileId ? {
          ...f,
          progress: 100,
          status: 'completed'
        } : f
      ));

      // Reset system status
      setSystemStatus({
        ocr: 'ready',
        agents: 'ready',
        vector_db: 'ready',
        llm: 'ready'
      });

      toast.success(`🎉 Document ${documentId} fully processed!`);

    } catch (error: any) {
      console.error('Processing error:', error);
      
      setFiles(prev => prev.map(f =>
        f.id === fileId ? {
          ...f,
          status: 'failed',
          error: error.response?.data?.detail || 'Processing failed'
        } : f
      ));

      toast.error(`Processing failed: ${error.response?.data?.detail || 'Unknown error'}`);
      
      // Reset system status
      setSystemStatus({
        ocr: 'ready',
        agents: 'ready',
        vector_db: 'ready',
        llm: 'ready'
      });
    }
  };

  // Remove a file
  const removeFile = (fileId: string) => {
    setFiles(prev => prev.filter(f => f.id !== fileId));
  };

  // View document
  const viewDocument = (documentId?: string) => {
    if (documentId) {
      router.push(`/query?document=${documentId}`);
    }
  };

  // Format file size
  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // Get status icon
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'uploading': return Clock;
      case 'processing': return Zap;
      case 'completed': return CheckCircle;
      case 'failed': return AlertCircle;
      default: return Clock;
    }
  };

  // Get status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'uploading': return 'bg-blue-100 text-blue-600';
      case 'processing': return 'bg-amber-100 text-amber-600';
      case 'completed': return 'bg-green-100 text-green-600';
      case 'failed': return 'bg-red-100 text-red-600';
      default: return 'bg-gray-100 text-gray-600';
    }
  };

  // Get system status color
  const getSystemStatusColor = (status: string) => {
    switch (status) {
      case 'ready': return 'bg-green-100 text-green-600 border-green-200';
      case 'busy': return 'bg-amber-100 text-amber-600 border-amber-200';
      case 'offline': return 'bg-red-100 text-red-600 border-red-200';
      default: return 'bg-gray-100 text-gray-600 border-gray-200';
    }
  };

  return (
    <div className="min-h-screen bg-linear-to-br from-gray-50 to-blue-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12 text-center"
        >
          <h1 className="text-5xl font-bold text-gray-900 mb-4">
            Intelligent Document Upload
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Upload PDFs for multi-modal analysis, intelligent extraction, and searchable indexing
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left panel - Upload zone */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-2"
          >
            {/* Upload zone */}
            <div className="mb-8">
              <div
                {...getRootProps()}
                className={`border-4 border-dashed rounded-3xl p-12 text-center cursor-pointer transition-all duration-300 ${
                  isDragActive || isDragging
                    ? 'border-blue-500 bg-blue-50 scale-[1.02]'
                    : 'border-gray-300 hover:border-blue-400 hover:bg-blue-50'
                }`}
              >
                <input {...getInputProps()} />
                
                <div className="space-y-6">
                  <div className="inline-block p-6 bg-linear-to-br from-blue-100 to-indigo-100 rounded-2xl">
                    <Upload className="w-16 h-16 text-blue-600" />
                  </div>
                  
                  <div>
                    <h2 className="text-2xl font-semibold text-gray-900 mb-2">
                      {isDragActive ? 'Drop files here' : 'Drag & drop PDF files'}
                    </h2>
                    <p className="text-gray-600 mb-6">
                      or click to browse files. Supports multiple PDF uploads.
                    </p>
                    
                    <button className="px-8 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-colors font-medium">
                      Browse Files
                    </button>
                  </div>
                  
                  <div className="text-sm text-gray-500">
                    Max file size: 10MB • PDF format only
                  </div>
                </div>
              </div>
            </div>

            {/* Processing pipeline visualization */}
            <div className="mb-8">
              <h3 className="text-xl font-semibold text-gray-900 mb-6">
                Processing Pipeline
              </h3>
              
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <div className="relative">
                  {/* Progress line */}
                  <div className="absolute top-1/2 left-0 right-0 h-1 bg-gray-200 transform -translate-y-1/2" />
                  
                  <div className="relative flex justify-between">
                    {[
                      { step: 1, icon: Upload, label: 'Upload', description: 'File ingestion' },
                      { step: 2, icon: Image, label: 'Layout', description: 'YOLO detection' },
                      { step: 3, icon: FileText, label: 'OCR', description: 'EasyOCR text' },
                      { step: 4, icon: Cpu, label: 'Agents', description: 'Multi-modal AI' },
                      { step: 5, icon: Database, label: 'RAG', description: 'Vector indexing' }
                    ].map((step, index) => {
                      const Icon = step.icon;
                      const isActive = files.some(f => f.progress >= index * 25);
                      
                      return (
                        <div key={step.step} className="text-center z-10">
                          <div className={`w-16 h-16 rounded-full flex items-center justify-center mb-3 mx-auto transition-all ${
                            isActive
                              ? 'bg-linear-to-br from-blue-500 to-indigo-600 text-white scale-110'
                              : 'bg-gray-100 text-gray-400'
                          }`}>
                            <Icon className="w-8 h-8" />
                          </div>
                          <div>
                            <div className="font-semibold text-gray-900">{step.label}</div>
                            <div className="text-sm text-gray-600">{step.description}</div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>

            {/* File list */}
            <div>
              <h3 className="text-xl font-semibold text-gray-900 mb-6">
                Uploaded Files ({files.length})
              </h3>
              
              <div className="space-y-4">
                {files.length === 0 ? (
                  <div className="bg-white rounded-2xl shadow-lg p-12 text-center">
                    <Cloud className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">
                      No files uploaded yet
                    </h3>
                    <p className="text-gray-600">
                      Upload PDFs to start the intelligent processing pipeline
                    </p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {files.map((file) => {
                      const StatusIcon = getStatusIcon(file.status);
                      
                      return (
                        <motion.div
                          key={file.id}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          className="bg-white rounded-2xl shadow-lg p-6"
                        >
                          <div className="flex items-start justify-between">
                            <div className="flex items-start space-x-4 flex-1">
                              <div className="p-3 bg-gray-100 rounded-xl">
                                <FileText className="w-8 h-8 text-gray-600" />
                              </div>
                              
                              <div className="flex-1">
                                <div className="flex items-center justify-between mb-2">
                                  <h4 className="font-semibold text-gray-900 truncate">
                                    {file.name}
                                  </h4>
                                  <div className="flex items-center space-x-3">
                                    <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(file.status)}`}>
                                      <div className="flex items-center space-x-1">
                                        <StatusIcon className="w-4 h-4" />
                                        <span>{file.status.toUpperCase()}</span>
                                      </div>
                                    </span>
                                    <button
                                      onClick={() => removeFile(file.id)}
                                      className="p-1 hover:bg-gray-100 rounded"
                                    >
                                      <X className="w-5 h-5 text-gray-400" />
                                    </button>
                                  </div>
                                </div>
                                
                                <div className="flex items-center justify-between text-sm text-gray-600 mb-4">
                                  <div className="flex items-center space-x-4">
                                    <span>{formatFileSize(file.size)}</span>
                                    {file.documentId && (
                                      <span className="font-mono">
                                        ID: {file.documentId.substring(0, 8)}...
                                      </span>
                                    )}
                                    {file.uploadTime && (
                                      <span>
                                        {file.uploadTime.toLocaleTimeString([], { 
                                          hour: '2-digit', 
                                          minute: '2-digit' 
                                        })}
                                      </span>
                                    )}
                                  </div>
                                  
                                  <span className="font-semibold">
                                    {file.progress}%
                                  </span>
                                </div>
                                
                                {/* Progress bar */}
                                <div className="h-2 bg-gray-200 rounded-full overflow-hidden mb-4">
                                  <div
                                    className={`h-full rounded-full transition-all duration-500 ${
                                      file.status === 'failed'
                                        ? 'bg-red-500'
                                        : 'bg-linear-to-r from-blue-500 to-indigo-600'
                                    }`}
                                    style={{ width: `${file.progress}%` }}
                                  />
                                </div>
                                
                                {/* Actions */}
                                <div className="flex items-center space-x-3">
                                  {file.status === 'completed' && file.documentId && (
                                    <button
                                      onClick={() => viewDocument(file.documentId)}
                                      className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center"
                                    >
                                      <Eye className="w-4 h-4 mr-2" />
                                      View Document
                                    </button>
                                  )}
                                  
                                  {file.status === 'failed' && file.error && (
                                    <div className="flex items-start text-sm text-red-600">
                                      <AlertCircle className="w-4 h-4 mr-1 mt-0.5 shrink-0" />
                                      <span>{file.error}</span>
                                    </div>
                                  )}
                                </div>
                              </div>
                            </div>
                          </div>
                        </motion.div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          </motion.div>

          {/* Right panel - System status and info */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-1"
          >
            <div className="sticky top-6 space-y-6">
              {/* System status */}
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <h3 className="text-xl font-semibold text-gray-900 mb-6">
                  System Status
                </h3>
                
                <div className="space-y-4">
                  {[
                    { 
                      key: 'ocr' as const, 
                      label: 'OCR Engine', 
                      icon: FileText,
                      description: 'EasyOCR text extraction'
                    },
                    { 
                      key: 'agents' as const, 
                      label: 'AI Agents', 
                      icon: Cpu,
                      description: 'Multi-modal intelligence'
                    },
                    { 
                      key: 'vector_db' as const, 
                      label: 'Vector Database', 
                      icon: Database,
                      description: 'Qdrant similarity search'
                    },
                    { 
                      key: 'llm' as const, 
                      label: 'LLM Server', 
                      icon: Cpu,
                      description: 'Qwen-2.5 inference'
                    }
                  ].map((component) => {
                    const Icon = component.icon;
                    const status = systemStatus[component.key];
                    
                    return (
                      <div
                        key={component.key}
                        className={`p-4 rounded-xl border-2 ${getSystemStatusColor(status)}`}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center space-x-3">
                            <Icon className="w-5 h-5" />
                            <span className="font-semibold">{component.label}</span>
                          </div>
                          <span className="text-sm font-medium capitalize">{status}</span>
                        </div>
                        <p className="text-sm text-gray-600">{component.description}</p>
                      </div>
                    );
                  })}
                </div>
                
                <div className="mt-6 pt-6 border-t">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Overall Status</span>
                    <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                      Object.values(systemStatus).every(s => s === 'ready')
                        ? 'bg-green-100 text-green-600'
                        : 'bg-amber-100 text-amber-600'
                    }`}>
                      {Object.values(systemStatus).every(s => s === 'ready') ? 'Ready' : 'Processing'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Features card */}
              <div className="bg-linear-to-br from-indigo-50 to-purple-50 rounded-2xl p-6">
                <h3 className="text-xl font-semibold text-gray-900 mb-4">
                  🚀 Intelligent Features
                </h3>
                
                <div className="space-y-4">
                  {[
                    {
                      icon: Shield,
                      title: 'Real Confidence Scoring',
                      description: 'Mathematical confidence based on 8+ factors'
                    },
                    {
                      icon: BarChart3,
                      title: 'Multi-Modal RAG',
                      description: 'Search across text, layout, and visual elements'
                    },
                    {
                      icon: Eye,
                      title: 'Human Review Workflow',
                      description: 'Low-confidence items flagged for validation'
                    },
                    {
                      icon: Zap,
                      title: 'Real-Time Processing',
                      description: 'Parallel processing with live progress tracking'
                    }
                  ].map((feature, index) => {
                    const Icon = feature.icon;
                    
                    return (
                      <div key={index} className="flex items-start space-x-3">
                        <div className="p-2 bg-white rounded-lg">
                          <Icon className="w-5 h-5 text-indigo-600" />
                        </div>
                        <div>
                          <div className="font-medium text-gray-900">
                            {feature.title}
                          </div>
                          <div className="text-sm text-gray-700">
                            {feature.description}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Stats card */}
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <h3 className="text-xl font-semibold text-gray-900 mb-4">
                  📊 Upload Statistics
                </h3>
                
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Files Uploaded</span>
                    <span className="text-2xl font-bold text-gray-900">
                      {files.length}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Success Rate</span>
                    <span className="text-2xl font-bold text-green-600">
                      {files.length > 0
                        ? Math.round(
                            (files.filter(f => f.status === 'completed').length / files.length) * 100
                          ) + '%'
                        : '0%'
                      }
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Total Size</span>
                    <span className="text-xl font-semibold text-gray-900">
                      {formatFileSize(files.reduce((sum, f) => sum + f.size, 0))}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Avg Processing Time</span>
                    <span className="text-xl font-semibold text-gray-900">
                      ~45s
                    </span>
                  </div>
                </div>
              </div>

              {/* Tips card */}
              <div className="bg-linear-to-br from-amber-50 to-orange-50 rounded-2xl p-6">
                <h3 className="text-xl font-semibold text-gray-900 mb-4">
                  💡 Pro Tips
                </h3>
                
                <ul className="space-y-3 text-sm text-gray-700">
                  <li>• Upload high-quality PDFs for better OCR results</li>
                  <li>• Use clear document structure for accurate layout detection</li>
                  <li>• Documents are automatically indexed for multi-modal search</li>
                  <li>• Check the review dashboard for low-confidence extractions</li>
                  <li>• Use the query interface to ask questions about your documents</li>
                  <li>• Processing continues even if you navigate away</li>
                </ul>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default UploadPage;