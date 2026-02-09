
__all__ = ['_patch_priority']

def _patch_priority(classes, priority_name, replaced_operators):
    """Patch new prioritys..
    
    This is necessary to use ``__r*__`` methods like ``__rmatmul__`` from custom classes
    instead of ``__*__`` methods like ``__matmul__`` from the given classes if the never return :const:`NotImplemented`.
    
    This function replaces operators like ``__matmul__`` or ``__eq__`` of
    classes. In the new operator it is first 
    checked if ``other`` has the attribute ``priority_name`` with a higher value than 
    the classes's ``priority_name`` which is set to 0.0. If so, :const:`NotImplemented`
    is returned. Otherwise the original implementation of the operator is called.

    This workaround is based on:
    https://github.com/scipy/scipy/issues/4819#issuecomment-920722279
    """
    
    def teach_priority(operator):
        def respect_priority(self, other):
            self_priority = getattr(self, priority_name, 0.0)
            other_priority = getattr(other, priority_name, -1.0)
            if self_priority < other_priority:
                return NotImplemented
            else:
                return operator(self, other)
        return respect_priority

    for c in classes:
        # Patch the internal _spbase class which is the actual base for all sparse types
        # This is where __matmul__ and _matmul_dispatch are defined
        if not hasattr(c, priority_name):
            setattr(c, priority_name, 0.0)
        
        for operator_name in replaced_operators:
            if hasattr(c, operator_name):
                operator = getattr(c, operator_name)
                check_if_applied = '__priority_patch_applied' + priority_name
                if not getattr(operator, check_if_applied, False):
                    wrapped_operator = teach_priority(operator)
                    setattr(wrapped_operator, check_if_applied, True)
                    setattr(c, operator_name, wrapped_operator)