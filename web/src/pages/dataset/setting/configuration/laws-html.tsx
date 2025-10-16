import { useEffect } from 'react';
import { useFormContext } from 'react-hook-form';

import { DocumentType } from '@/components/layout-recognize-form-field';

import { LawsConfiguration } from './laws';

export function LawsHtmlConfiguration() {
  const form = useFormContext();

  useEffect(() => {
    const current = form.getValues('parser_config.layout_recognize');
    if (
      !current ||
      current === DocumentType.DeepDocVN ||
      current === DocumentType.DeepDOC
    ) {
      form.setValue(
        'parser_config.layout_recognize',
        DocumentType.DeepDocHTML,
        {
          shouldValidate: false,
          shouldDirty: false,
        },
      );
    }
  }, [form]);

  return <LawsConfiguration />;
}
