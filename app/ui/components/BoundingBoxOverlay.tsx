import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FileText, Table, Image as ImageIcon, PenTool as Signature, X } from 'lucide-react';

interface BoundingBoxOverlayProps {
  regions: Array<{
    id: string;
    type: 'text_block' | 'table' | 'figure' | 'signature';
    bbox: { x1: number; y1: number; x2: number; y2: number };
    confidence: number;
    text?: string;
  }>;
  containerWidth: number;
  containerHeight: number;
  onRegionClick?: (region: any) => void;
}

const BoundingBoxOverlay: React.FC<BoundingBoxOverlayProps> = ({
  regions,
  containerWidth,
  containerHeight,
  onRegionClick
}) => {
  const [selectedRegion, setSelectedRegion] = useState<string | null>(null);
  const [hoveredRegion, setHoveredRegion] = useState<string | null>(null);

  const getRegionColor = (type: string) => {
    switch (type) {
      case 'text_block': return 'border-blue-500 bg-blue-500/10';
      case 'table': return 'border-green-500 bg-green-500/10';
      case 'figure': return 'border-purple-500 bg-purple-500/10';
      case 'signature': return 'border-amber-500 bg-amber-500/10';
      default: return 'border-gray-500 bg-gray-500/10';
    }
  };

  const getRegionIcon = (type: string) => {
    switch (type) {
      case 'text_block': return FileText;
      case 'table': return Table;
      case 'figure': return ImageIcon;
      case 'signature': return Signature;
      default: return FileText;
    }
  };

  const handleRegionClick = (region: any) => {
    setSelectedRegion(region.id === selectedRegion ? null : region.id);
    if (onRegionClick) {
      onRegionClick(region);
    }
  };

  const handleCloseDetails = (e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedRegion(null);
  };

  return (
    <>
      {/* Bounding boxes */}
      {regions.map((region) => {
        const width = (region.bbox.x2 - region.bbox.x1) * containerWidth;
        const height = (region.bbox.y2 - region.bbox.y1) * containerHeight;
        const left = region.bbox.x1 * containerWidth;
        const top = region.bbox.y2 * containerHeight;

        const Icon = getRegionIcon(region.type);
        const colorClass = getRegionColor(region.type);
        const isSelected = selectedRegion === region.id;
        const isHovered = hoveredRegion === region.id;

        return (
          <React.Fragment key={region.id}>
            {/* Bounding box */}
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.2 }}
              className={`absolute border-2 rounded cursor-pointer ${colorClass} ${
                isSelected ? 'ring-2 ring-offset-2 ring-blue-500 z-50' : ''
              } ${isHovered ? 'ring-1 ring-offset-1 ring-current' : ''}`}
              style={{
                width,
                height,
                left,
                top: top - height,
                transform: 'translateY(-100%)'
              }}
              onClick={() => handleRegionClick(region)}
              onMouseEnter={() => setHoveredRegion(region.id)}
              onMouseLeave={() => setHoveredRegion(null)}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              {/* Region label */}
              <div className="absolute -top-6 left-0 flex items-center space-x-1">
                <Icon className="w-3 h-3" />
                <span className="text-xs font-medium whitespace-nowrap">
                  {region.type.replace('_', ' ')}
                </span>
                <span className="text-xs opacity-75">
                  {Math.round(region.confidence * 100)}%
                </span>
              </div>
            </motion.div>

            {/* Region details popup */}
            <AnimatePresence>
              {isSelected && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 10 }}
                  className="absolute z-50 bg-white rounded-lg shadow-xl border p-4 max-w-sm"
                  style={{
                    left: Math.min(left + width + 10, containerWidth - 320),
                    top: Math.max(top - height - 10, 10)
                  }}
                  onClick={(e) => e.stopPropagation()}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center space-x-2">
                      <Icon className="w-5 h-5" />
                      <h3 className="font-semibold capitalize">
                        {region.type.replace('_', ' ')}
                      </h3>
                    </div>
                    <button
                      onClick={handleCloseDetails}
                      className="p-1 hover:bg-gray-100 rounded"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>

                  <div className="space-y-3">
                    <div>
                      <div className="text-sm text-gray-600 mb-1">Confidence</div>
                      <div className="flex items-center space-x-2">
                        <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-blue-500 rounded-full"
                            style={{ width: `${region.confidence * 100}%` }}
                          />
                        </div>
                        <span className="text-sm font-medium">
                          {Math.round(region.confidence * 100)}%
                        </span>
                      </div>
                    </div>

                    <div>
                      <div className="text-sm text-gray-600 mb-1">Position</div>
                      <div className="text-sm">
                        ({Math.round(region.bbox.x1 * 100)}%, {Math.round(region.bbox.y1 * 100)}%) to
                        ({Math.round(region.bbox.x2 * 100)}%, {Math.round(region.bbox.y2 * 100)}%)
                      </div>
                    </div>

                    {region.text && (
                      <div>
                        <div className="text-sm text-gray-600 mb-1">Extracted Text</div>
                        <div className="text-sm bg-gray-50 p-2 rounded">
                          {region.text.length > 200
                            ? `${region.text.substring(0, 200)}...`
                            : region.text}
                        </div>
                      </div>
                    )}

                    <div className="pt-2 border-t">
                      <button
                        className="w-full py-2 text-sm bg-blue-50 text-blue-600 rounded hover:bg-blue-100 transition-colors"
                        onClick={() => handleRegionClick(region)}
                      >
                        View Full Details
                      </button>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </React.Fragment>
        );
      })}

      {/* Selection overlay */}
      {selectedRegion && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setSelectedRegion(null)}
        />
      )}
    </>
  );
};

export default BoundingBoxOverlay;