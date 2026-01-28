import React, { useState, useEffect, useRef } from 'react';
import Image from 'next/image';
import { motion } from 'framer-motion';
import { ZoomIn, ZoomOut, RotateCw, Download, Eye, EyeOff } from 'lucide-react';
import BoundingBoxOverlay from './BoundingBoxOverlay';
import ConfidenceMeter from './ConfidenceMeter';
import AgentTracePanel from './AgentTracePanel';

interface DocumentViewerProps {
  documentId: string;
  pages: Array<{
    pageNumber: number;
    imageUrl: string;
    width: number;
    height: number;
    regions?: Array<{
      id: string;
      type: 'text_block' | 'table' | 'figure' | 'signature';
      bbox: { x1: number; y1: number; x2: number; y2: number };
      confidence: number;
      text?: string;
    }>;
  }>;
  // Change the agentTraces prop type:
agentTraces?: Array<{
  agent: string;
  confidence: number;
  timestamp: string;
  insights: string[];
  status?: 'completed' | 'processing' | 'failed';  // Make optional
}>
  onRegionClick?: (region: any) => void;
  className?: string;
}

const DocumentViewer: React.FC<DocumentViewerProps> = ({
  documentId,
  pages,
  agentTraces = [],
  onRegionClick,
  className = ''
}) => {
  const [currentPage, setCurrentPage] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [showRegions, setShowRegions] = useState(true);
  const [showAgentTraces, setShowAgentTraces] = useState(true);
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  const currentPageData = pages[currentPage];

  // Zoom controls
  const handleZoomIn = () => setZoom(prev => Math.min(prev + 0.25, 3));
  const handleZoomOut = () => setZoom(prev => Math.max(prev - 0.25, 0.5));
  const handleZoomReset = () => setZoom(1);

  // Navigation
  const handlePrevPage = () => {
    if (currentPage > 0) {
      setCurrentPage(prev => prev - 1);
    }
  };

  const handleNextPage = () => {
    if (currentPage < pages.length - 1) {
      setCurrentPage(prev => prev + 1);
    }
  };

  // Calculate image dimensions
  const maxWidth = 800;
  const aspectRatio = currentPageData.width / currentPageData.height;
  const displayWidth = Math.min(maxWidth, currentPageData.width * zoom);
  const displayHeight = displayWidth / aspectRatio;

  // Agent confidence stats
  const avgConfidence = agentTraces.length > 0
    ? agentTraces.reduce((sum, trace) => sum + trace.confidence, 0) / agentTraces.length
    : 0;

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Toolbar */}
      <div className="flex items-center justify-between p-4 border-b bg-white/80 backdrop-blur-sm">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <button
              onClick={handleZoomOut}
              className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
              title="Zoom out"
            >
              <ZoomOut className="w-5 h-5" />
            </button>
            <span className="text-sm font-medium w-16 text-center">
              {Math.round(zoom * 100)}%
            </span>
            <button
              onClick={handleZoomIn}
              className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
              title="Zoom in"
            >
              <ZoomIn className="w-5 h-5" />
            </button>
            <button
              onClick={handleZoomReset}
              className="p-2 rounded-lg hover:bg-gray-100 transition-colors ml-2"
              title="Reset zoom"
            >
              <RotateCw className="w-5 h-5" />
            </button>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowRegions(!showRegions)}
              className={`p-2 rounded-lg transition-colors ${showRegions ? 'bg-blue-100 text-blue-600' : 'hover:bg-gray-100'}`}
              title={showRegions ? 'Hide regions' : 'Show regions'}
            >
              {showRegions ? <Eye className="w-5 h-5" /> : <EyeOff className="w-5 h-5" />}
            </button>
            <span className="text-sm">Regions</span>
          </div>
        </div>

        <div className="flex items-center space-x-6">
          {/* Page navigation */}
          <div className="flex items-center space-x-3">
            <button
              onClick={handlePrevPage}
              disabled={currentPage === 0}
              className={`px-3 py-1 rounded-lg ${currentPage === 0 ? 'opacity-50 cursor-not-allowed' : 'hover:bg-gray-100'}`}
            >
              Previous
            </button>
            <span className="text-sm font-medium">
              Page {currentPage + 1} of {pages.length}
            </span>
            <button
              onClick={handleNextPage}
              disabled={currentPage === pages.length - 1}
              className={`px-3 py-1 rounded-lg ${currentPage === pages.length - 1 ? 'opacity-50 cursor-not-allowed' : 'hover:bg-gray-100'}`}
            >
              Next
            </button>
          </div>

          {/* Confidence display */}
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-600">Confidence:</span>
            <ConfidenceMeter confidence={avgConfidence} size="sm" />
            <span className="text-sm font-medium">
              {Math.round(avgConfidence * 100)}%
            </span>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Document view */}
        <div className="flex-1 overflow-auto p-6" ref={containerRef}>
          <div className="flex justify-center">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3 }}
              className="relative bg-white rounded-xl shadow-lg overflow-hidden"
              style={{ width: displayWidth, height: displayHeight }}
            >
              {/* Document image */}
              <div className="relative w-full h-full">
                <img
                  ref={imageRef}
                  src={currentPageData.imageUrl}
                  alt={`Page ${currentPage + 1}`}
                  className="w-full h-full object-contain"
                />

                {/* Bounding boxes overlay */}
                {showRegions && currentPageData.regions && (
                  <BoundingBoxOverlay
                    regions={currentPageData.regions}
                    containerWidth={displayWidth}
                    containerHeight={displayHeight}
                    onRegionClick={onRegionClick}
                  />
                )}

                {/* Page number indicator */}
                <div className="absolute bottom-4 right-4 bg-black/70 text-white px-3 py-1 rounded-full text-sm">
                  Page {currentPage + 1}
                </div>
              </div>
            </motion.div>
          </div>

          {/* Region statistics */}
          {currentPageData.regions && (
            <div className="mt-6 max-w-2xl mx-auto">
              <div className="grid grid-cols-4 gap-4">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">
                    {currentPageData.regions.filter(r => r.type === 'text_block').length}
                  </div>
                  <div className="text-sm text-blue-800">Text Blocks</div>
                </div>
                <div className="bg-green-50 p-4 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">
                    {currentPageData.regions.filter(r => r.type === 'table').length}
                  </div>
                  <div className="text-sm text-green-800">Tables</div>
                </div>
                <div className="bg-purple-50 p-4 rounded-lg">
                  <div className="text-2xl font-bold text-purple-600">
                    {currentPageData.regions.filter(r => r.type === 'figure').length}
                  </div>
                  <div className="text-sm text-purple-800">Figures</div>
                </div>
                <div className="bg-amber-50 p-4 rounded-lg">
                  <div className="text-2xl font-bold text-amber-600">
                    {currentPageData.regions.filter(r => r.type === 'signature').length}
                  </div>
                  <div className="text-sm text-amber-800">Signatures</div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Agent traces panel */}
        {showAgentTraces && agentTraces.length > 0 && (
          <div className="w-96 border-l overflow-auto">
            <AgentTracePanel traces={agentTraces} />
          </div>
        )}
      </div>
    </div>
  );
};

export default DocumentViewer;