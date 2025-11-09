import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import { cn } from '@/lib/utils';
import { CollapsibleProps } from '@radix-ui/react-collapsible';
import {
  ChevronDown,
  ChevronUp,
  ListChevronsDownUp,
  ListChevronsUpDown,
} from 'lucide-react';
import * as React from 'react';
import {
  PropsWithChildren,
  ReactNode,
  useCallback,
  useEffect,
  useState,
} from 'react';

type CollapseProps = Omit<CollapsibleProps, 'title'> & {
  title?: ReactNode;
  rightContent?: ReactNode;
} & PropsWithChildren;

export function Collapse({
  title,
  children,
  rightContent,
  open = true,
  defaultOpen = false,
  onOpenChange,
  disabled,
}: CollapseProps) {
  const [currentOpen, setCurrentOpen] = useState(open);

  useEffect(() => {
    setCurrentOpen(open);
  }, [open]);

  const handleOpenChange = useCallback(
    (open: boolean) => {
      setCurrentOpen(open);
      onOpenChange?.(open);
    },
    [onOpenChange],
  );

  return (
    <Collapsible
      defaultOpen={defaultOpen}
      open={currentOpen}
      onOpenChange={handleOpenChange}
      disabled={disabled}
    >
      <CollapsibleTrigger className={'w-full'}>
        <section className="flex justify-between items-center">
          <div className="flex items-center gap-1">
            {currentOpen ? (
              <ListChevronsUpDown className="size-4" />
            ) : (
              <ListChevronsDownUp className="size-4 text-text-secondary" />
            )}
            <div
              className={cn('text-text-secondary', {
                'text-text-primary': open,
              })}
            >
              {title}
            </div>
          </div>
          <div>{rightContent}</div>
        </section>
      </CollapsibleTrigger>
      <CollapsibleContent className="pt-5">{children}</CollapsibleContent>
    </Collapsible>
  );
}

export type NodeCollapsibleProps<T extends any[]> = {
  items?: T;
  children: (item: T[0], idx: number) => ReactNode;
  className?: string;
};
export function NodeCollapsible<T extends any[]>({
  items = [] as unknown as T,
  children,
  className,
}: NodeCollapsibleProps<T>) {
  const [isOpen, setIsOpen] = React.useState(false);

  const nextClassName = cn('space-y-2', className);

  const baseItems = React.useMemo(() => {
    return Array.isArray(items) ? [...items] : ([] as unknown as T);
  }, [items]);

  const nextItems = React.useMemo(() => {
    return baseItems.every((x) => Array.isArray(x))
      ? (baseItems.flat() as Array<T[number]>)
      : (baseItems as Array<T[number]>);
  }, [baseItems]);

  const getItemKey = React.useCallback((item: T[number], index: number) => {
    if (item == null) {
      return index;
    }

    if (typeof item === 'string' || typeof item === 'number') {
      return `${item}`;
    }

    if (typeof item === 'object') {
      const possibleKey =
        // @ts-expect-error index access
        (item.id as string | number | undefined) ??
        // @ts-expect-error index access
        item.mcp_id ??
        // @ts-expect-error index access
        item.component_name ??
        undefined;

      if (possibleKey !== undefined) {
        return `${possibleKey}`;
      }
    }

    return index;
  }, []);

  const renderItems = React.useCallback(
    (list: Array<T[number]>, startIndex = 0) =>
      list.map((item, idx) => {
        const element = children(item, idx + startIndex);
        const key = getItemKey(item, idx + startIndex);

        if (React.isValidElement(element)) {
          return element.key != null
            ? element
            : React.cloneElement(element, { key });
        }

        return <React.Fragment key={key}>{element}</React.Fragment>;
      }),
    [children, getItemKey],
  );

  return (
    <Collapsible
      open={isOpen}
      onOpenChange={setIsOpen}
      className={cn('relative', nextClassName)}
    >
      {renderItems(nextItems.slice(0, 3))}
      <CollapsibleContent className={nextClassName}>
        {renderItems(nextItems.slice(3), 3)}
      </CollapsibleContent>
      {nextItems.length > 3 && (
        <CollapsibleTrigger
          asChild
          onClick={(e) => e.stopPropagation()}
          className="absolute left-1/2 -translate-x-1/2 bottom-0 translate-y-1/2 cursor-pointer"
        >
          <div
            className={cn(
              'size-3 bg-text-secondary rounded-full flex items-center justify-center',
              { 'bg-text-primary': isOpen },
            )}
          >
            {isOpen ? (
              <ChevronUp className="stroke-bg-component" />
            ) : (
              <ChevronDown className="stroke-bg-component" />
            )}
          </div>
        </CollapsibleTrigger>
      )}
    </Collapsible>
  );
}
