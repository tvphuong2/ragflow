import { ElementDatum, Graph, IElementEvent } from '@antv/g6';
import isEmpty from 'lodash/isEmpty';
import { useCallback, useEffect, useMemo, useRef } from 'react';
import { buildNodesAndCombos } from './util';

import styles from './index.less';

const TooltipColorMap = {
  combo: 'red',
  node: 'black',
  edge: 'blue',
};

function safeJSONParse(value: string) {
  try {
    return JSON.parse(value);
  } catch (error) {
    console.error('[ForceGraph] Failed to parse graph payload', error);
    return {};
  }
}

interface IProps {
  data: any;
  show: boolean;
}

const ForceGraph = ({ data, show }: IProps) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<Graph | null>(null);

  const nextData = useMemo(() => {
    if (isEmpty(data)) {
      return { nodes: [], edges: [], combos: [], hasCombos: false };
    }

    const parsed = typeof data === 'string' ? safeJSONParse(data) : data;
    const nodes = Array.isArray(parsed?.nodes) ? parsed.nodes : [];
    const edges = Array.isArray(parsed?.edges) ? parsed.edges : [];
    const { nodes: nextNodes, combos, hasCombos } = buildNodesAndCombos(nodes);

    return { edges, nodes: nextNodes, combos, hasCombos };
  }, [data]);

  const render = useCallback(() => {
    const { nodes, edges, combos, hasCombos } = nextData;
    const behaviors = [
      'drag-element',
      'drag-canvas',
      'zoom-canvas',
      ...(hasCombos ? ['collapse-expand'] : []),
      {
        type: 'hover-activate',
        degree: 1, // 👈🏻 Activate relations.
      },
    ];

    const layout = hasCombos
      ? {
          type: 'combo-combined' as const,
          preventOverlap: true,
          comboPadding: 24,
          spacing: 160,
          nodeSpacing: 80,
        }
      : {
          type: 'force2' as const,
          preventOverlap: true,
          linkDistance: 240,
          nodeStrength: -200,
          edgeStrength: 0.2,
          minMovement: 0.1,
          maxIteration: 600,
        };

    const graph = new Graph({
      container: containerRef.current!,
      autoFit: 'view',
      autoResize: true,
      behaviors,
      plugins: [
        {
          type: 'tooltip',
          enterable: true,
          getContent: (e: IElementEvent, items: ElementDatum) => {
            if (Array.isArray(items)) {
              if (items.some((x) => x?.isCombo)) {
                return `<p style="font-weight:600;color:red">${items?.[0]?.data?.label}</p>`;
              }
              let result = ``;
              items.forEach((item) => {
                result += `<section style="color:${TooltipColorMap[e['targetType'] as keyof typeof TooltipColorMap]};"><h3>${item?.id}</h3>`;
                if (item?.entity_type) {
                  result += `<div style="padding-bottom: 6px;"><b>Entity type: </b>${item?.entity_type}</div>`;
                }
                if (item?.weight) {
                  result += `<div><b>Weight: </b>${item?.weight}</div>`;
                }
                if (item?.description) {
                  result += `<p>${item?.description}</p>`;
                }
              });
              return result + '</section>';
            }
            return undefined;
          },
        },
      ],
      layout,
      node: {
        style: {
          size: 96,
          labelText: (d) => d.id,
          labelFontSize: 22,
          labelOffsetY: 12,
          labelPlacement: 'center',
          labelWordWrap: true,
        },
        palette: {
          type: 'group',
          field: (d) => {
            return d?.entity_type as string;
          },
        },
      },
      edge: {
        style: (model) => {
          const weight: number = Number(model?.weight) || 2;
          const lineWeight = weight * 4;
          return {
            stroke: '#99ADD1',
            lineWidth: lineWeight > 10 ? 10 : lineWeight,
          };
        },
      },
    });

    if (graphRef.current) {
      graphRef.current.destroy();
    }

    graphRef.current = graph;

    graph.setData({ nodes, edges, combos });

    graph.render();
    graph.fitView(16);
  }, [nextData]);

  useEffect(() => {
    if (!containerRef.current) {
      return;
    }

    if (isEmpty(nextData.nodes) && isEmpty(nextData.edges)) {
      graphRef.current?.destroy();
      graphRef.current = null;
      return;
    }

    render();

    return () => {
      graphRef.current?.destroy();
      graphRef.current = null;
    };
  }, [nextData, render]);

  return (
    <div
      ref={containerRef}
      className={styles.forceContainer}
      style={{
        width: '100%',
        height: '100%',
        display: show ? 'block' : 'none',
      }}
    />
  );
};

export default ForceGraph;
